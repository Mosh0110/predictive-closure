import csv
import json
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from jepa.artifacts import sha256
from run import P_D, Q, SIZES, BETAS

COLORS = dict(A='#0072B2', B='#009E73', C='#E66101', D='#CC79A7',
              current_light='#4D4D4D', future_light='#8255A5')
MARKERS = dict(A='o', B='^', C='s', D='D')
LABELS = dict(AB='A/B set', CD='C/D set', current_light='Current light',
              future_light='Future light', class_A='Image A', class_B='Image B')
P_LABEL = r'$C\ \rightarrow\ D$ copy probability, $p_D$'
WEIGHTS = (.05,.10,.20)


def read(path):
    return json.loads(path.read_text())


def verified_result(directory, name, stage):
    path = directory/name
    assert sha256(path) == read(directory/f'complete_{stage}.json')[name], path
    return read(path)


def collect(root):
    rows = []
    for job in read(root/'plan.json'):
        directory = root/job['tag']
        stages = 3 if job['study'] == 'traffic' else 2
        if not all((directory/f'complete_{i}.json').exists() for i in range(stages)):
            continue
        result = verified_result(directory, 'probes.json', 1)
        if job['study'] == 'traffic':
            values = {k:v['test']['accuracy'] for k,v in result['probe_results'].items()}
        else:
            tests = result['evaluations']['test']
            names = sorted({r['position'] for r in tests})
            values = {name:mean(r['accuracy'] for r in tests if r['position'] == name) for name in names}
            for group, members in [('AB', ['A','B']), ('CD', [n for n in names if n not in ('A','B')])]:
                values[group] = mean(values[n] for n in members)
                values[group+'_min'] = min(values[n] for n in members)
                values[group+'_max'] = max(values[n] for n in members)
        assert all(0 <= value <= 1 for value in values.values())
        rows.append({**job, 'values':values})
    return rows


def points(rows, study, weight):
    selected = sorted((r for r in rows if r['study'] == study and r['weight'] == weight),
                      key=lambda r:(r['beta'],r['x']))
    xs = Q if study == 'traffic' else SIZES if study == 'subset' else P_D
    expected = {(beta,x) for beta in (BETAS if study == 'reconstruction' else (0.,)) for x in xs}
    assert len({(r['beta'],r['x']) for r in selected}) == len(selected)
    return selected if {(r['beta'],r['x']) for r in selected} == expected else []


def accuracy_axis(ax, selected, features, xlabel, categorical=False, chance=50, legend=True):
    xs = list(range(len(selected))) if categorical else [r['x'] for r in selected]
    for feature in features:
        color_key = 'A' if feature.startswith('AB') or feature == 'class_A' else 'C' if feature.startswith('CD') else 'B' if feature == 'class_B' else feature
        ax.plot(xs, [100*r['values'][feature] for r in selected],
                marker=MARKERS.get(color_key, 'o'), color=COLORS[color_key],
                label=LABELS.get(feature, feature))
    ax.axhline(chance, ls='--', color='0.25', lw=.8)
    ax.set_xticks(xs, [f'{r["x"]:.2f}' if categorical else f'{r["x"]:g}' for r in selected])
    values = [100*r['values'][f] for r in selected for f in features]+[chance]
    ax.set_ylim(max(0, min(values)-3), min(101, max(values)+5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Decoding accuracy (%)')
    ax.grid(axis='y', color='0.9', lw=.5)
    if legend:
        ax.legend(frameon=False, fontsize=8)


def reconstruction_figure(rows, weight):
    selected = points(rows, 'predictability', weight)+points(rows, 'reconstruction', weight)
    cells = {(r['beta'],r['x']):r['values'] for r in selected}
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3), layout='constrained')
    for ax, name in zip(axes, 'ABCD', strict=True):
        values = [[100*cells[beta,x][name] for x in P_D] for beta in (0.,*BETAS)]
        cmap = LinearSegmentedColormap.from_list(name, ['white', COLORS[name]])
        ax.imshow(values, origin='lower', vmin=50, vmax=90, cmap=cmap, aspect='auto')
        for i, line in enumerate(values):
            for j, value in enumerate(line):
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if value >= 79 else '#111111')
        ax.set_xticks(range(len(P_D)), [f'{x:.2f}' for x in P_D], rotation=45)
        ax.set_yticks(range(5), [f'{b:g}' for b in (0.,*BETAS)])
        ax.set_title(name, color=COLORS[name])
        ax.tick_params(length=0)
    axes[0].set_ylabel(r'Reconstruction weight $\beta$')
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.supxlabel(P_LABEL, fontsize=10)
    fig.suptitle(rf'Reconstruct C only; $r={weight:g}$ (color scale: 50-90%)', fontsize=11)
    return fig


def side_figure(rows, weight):
    selected = points(rows, 'side_input', weight)
    control = points(rows, 'predictability', weight)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), layout='constrained', sharey=True)
    values = [100*r['values'][f] for r in selected+control for f in 'ABCD']+[50]
    for ax, feature in zip(axes, 'ABCD', strict=True):
        accuracy_axis(ax, selected, [feature], '', categorical=True, legend=False)
        ax.lines[0].set_label('C supplied')
        ax.plot(range(5), [100*r['values'][feature] for r in control], 's--',
                color='#72777F', label='No side input')
        ax.set_ylim(max(0,min(values)-3), min(101,max(values)+5))
        ax.set_title(feature, color=COLORS[feature])
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelrotation=45)
    axes[0].set_ylabel('Decoding accuracy (%)')
    axes[-1].legend(frameon=False, fontsize=8)
    fig.supxlabel(P_LABEL, fontsize=10)
    fig.suptitle(rf'Supply C to the predictor; $r={weight:g}$', fontsize=11)
    return fig


def plot(root):
    rows = collect(root)
    assert rows, 'No completed experiments to plot.'
    output = root/'figures'
    output.mkdir(exist_ok=True)
    with (root/'accuracies.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('study','r','seed','x','beta','feature','accuracy'))
        for row in rows:
            for feature, accuracy in row['values'].items():
                writer.writerow((row['study'],row['weight'],row['seed'],row['x'],row['beta'],feature,accuracy))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'pdf.fonttype':42, 'lines.linewidth':1.8,
                         'lines.markersize':5, 'axes.axisbelow':True})
    files = []

    def save(fig, name):
        path = output/name
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        files.append(name)

    for study, features, xlabel, name in [
        ('traffic', ['current_light'], r'Light-control probability $q$', 'traffic_light'),
        ('predictability', list('ABCD'), P_LABEL, 'predictability'),
        ('subset', ['AB','CD'], r'Subset size $N$', 'subset_size'),
    ]:
        selected = points(rows, study, .10)
        if selected:
            fig, ax = plt.subplots(figsize=(4.8, 3.2), layout='constrained')
            accuracy_axis(ax, selected, features, xlabel, categorical=study == 'predictability')
            save(fig, f'main/{name}')
    for study, builder, name in [('reconstruction', reconstruction_figure, 'reconstruction'),
                                  ('side_input', side_figure, 'side_information')]:
        if points(rows, study, .10) and points(rows, 'predictability', .10):
            save(builder(rows, .10), f'main/{name}')
        if all(points(rows, study, r) and points(rows, 'predictability', r) for r in WEIGHTS):
            for weight in WEIGHTS:
                save(builder(rows, weight), f'sensitivity/{study}_r{round(weight*1000):03d}')
    subset = points(rows, 'subset', .10)
    if subset:
        fig, axes = plt.subplots(1, 2, figsize=(8,3.1), layout='constrained', sharey=True)
        values = [100*r['values'][f'{g}_{k}'] for r in subset for g in ('AB','CD') for k in ('min','max')]+[50]
        for ax, kind in zip(axes, ('min','max'), strict=True):
            accuracy_axis(ax, subset, [f'AB_{kind}',f'CD_{kind}'], r'Subset size $N$')
            ax.set_ylim(max(0,min(values)-3), min(101,max(values)+5))
            ax.set_title('Worst position' if kind == 'min' else 'Best position')
            ax.legend(handles=ax.lines[:2], labels=['A/B set','C/D set'], frameon=False, fontsize=8)
        save(fig, 'appendix/subset_extrema')
    traffic = points(rows, 'traffic', .10)
    if traffic:
        fig, axes = plt.subplots(1, 2, figsize=(8.8,3.2), layout='constrained')
        train_final, eval_final = [], []
        for row in traffic:
            result = verified_result(root/row['tag'], 'jepa.json', 0)
            curve = result['learning_curve']
            axes[0].plot([p['step'] for p in curve], [p['prediction_mse'] for p in curve], label=f'q={row["x"]:g}')
            train_final.append(curve[-1]['prediction_mse'])
            eval_final.append(result['evaluation_curve'][-1]['evaluation']['prediction_mse'])
        axes[1].plot(Q, train_final, 'o-', label='Training')
        axes[1].plot(Q, eval_final, 's-', label='Evaluation')
        for ax, label in zip(axes, ('Training step',r'Light-control probability $q$'), strict=True):
            ax.set_xlabel(label)
            ax.set_ylabel('Latent prediction MSE')
            ax.legend(frameon=False, fontsize=8)
        save(fig, 'appendix/traffic_prediction_loss')
    for study, groups in [
        ('traffic', [('traffic_light',['current_light','future_light'],50), ('traffic_images',['class_A','class_B'],10)]),
        ('predictability', [('predictability',list('ABCD'),50)]),
        ('subset', [('subset_size',['AB','CD'],50)]),
    ]:
        if not all(points(rows, study, r) for r in WEIGHTS):
            continue
        for name, features, chance in groups:
            fig, axes = plt.subplots(1, 3, figsize=(11.7,3.2), layout='constrained', sharey=True)
            for ax, weight in zip(axes, WEIGHTS, strict=True):
                selected = points(rows, study, weight)
                accuracy_axis(ax, selected, features, r'Light-control probability $q$' if study == 'traffic' else r'Subset size $N$' if study == 'subset' else P_LABEL,
                              categorical=study == 'predictability', chance=chance)
                ax.set_title(rf'$r={weight:g}$')
            values = [100*r['values'][f] for r in rows if r['study'] == study for f in features]+[chance]
            axes[0].set_ylim(max(0,min(values)-3), min(101,max(values)+5))
            save(fig, f'sensitivity/{name}')
    if all(points(rows, 'traffic', r) for r in WEIGHTS):
        fig, axes = plt.subplots(1, 3, figsize=(11.7,3.2), layout='constrained', sharey=True)
        for ax, weight in zip(axes, WEIGHTS, strict=True):
            deltas = []
            for row in points(rows, 'traffic', weight):
                values = verified_result(root/row['tag'], 'intervention.json', 2)['mean']
                deltas.append(values['counterfactual_prediction_mse']-values['original_prediction_mse'])
            ax.plot(Q, deltas, 'o-', color=COLORS['current_light'])
            ax.axhline(0, ls='--', color='0.25', lw=.8)
            ax.set_xlabel(r'Light-control probability $q$')
            ax.set_ylabel('Increase in prediction MSE')
            ax.set_title(rf'$r={weight:g}$')
        save(fig, 'sensitivity/traffic_intervention')
    planned = len(read(root/'plan.json'))
    cards = '\n'.join(f'<section><h2>{name}</h2><a href="{name}.pdf"><img src="{name}.png" alt="{name}"></a></section>' for name in files)
    (output/'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Predictive Closure figures</title>'
        '<style>body{font:16px sans-serif;max-width:1150px;margin:40px auto;padding:0 20px}'
        'section{margin:40px 0}img{max-width:100%}h2{font-size:18px}</style>'
        f'<h1>Predictive Closure</h1><p>{len(rows)}/{planned} encoder settings complete. '
        'Click a plot for its PDF. '
        'Figures appear when their parameter sweep is complete.</p>'+cards+'</html>')
    assert files, 'No complete parameter sweep to plot yet.'
    return files
