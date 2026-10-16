import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_SEED = 303
STUDIES = ('traffic', 'predictability', 'subset', 'reconstruction', 'side_input')
P_D = (.60, .75, .85, .95, 1.)
Q = (0., .25, .5, .75, 1.)
SIZES = (2, 4, 8, 16)
BETAS = (.0425, .105, .135, .36)


def template(name):
    return json.loads((ROOT/'configs'/f'{name}.json').read_text())


def jobs(studies, weights, seed, data, output, device='cuda:0'):
    records = []
    for study in STUDIES:
        if study not in studies:
            continue
        for weight in weights:
            values = Q if study == 'traffic' else SIZES if study == 'subset' else P_D
            for x in values:
                for beta in BETAS if study == 'reconstruction' else (0.,):
                    tag = f'{study}_r{weight:g}_x{x:g}_b{beta:g}_s{seed}'.replace('.', 'p')
                    directory = output/tag
                    train = template('traffic_train' if study == 'traffic' else 'subset_train' if study == 'subset' else 'four_image_train')
                    train.update(seed=seed, prediction_weight=1-weight, sigreg_weight=weight,
                                 run_id=tag, data_root=str(data), device=device,
                                 output_checkpoint=str(directory/'jepa.pt'), output_json=str(directory/'jepa.json'))
                    if study == 'traffic':
                        train['feed']['visible_mode_probability'] = x
                        train_module = 'jepa.traffic.train_jepa'
                        probe_module = 'jepa.traffic.probes'
                        probe = template('traffic_probe')
                    elif study == 'subset':
                        train['feed'].update(bd_ring_size=x, bd_render_tile_size={2:90,4:64,8:45,16:32}[x],
                                             bd_copy_probabilities=[.90,.96]*(x//2))
                        train_module = 'jepa.subset.train_jepa'
                        probe_module = 'jepa.image_probe'
                        probe = template('image_probe')
                    else:
                        train['feed']['probability_d'] = x
                        train['render_layout'] = 'columns'
                        train['training_corpus'] = str(output/'corpora'/f'pd{x:g}.pt')
                        train_module = 'jepa.four_image.train_jepa'
                        probe_module = 'jepa.image_probe'
                        probe = template('image_probe')
                        if study in ('reconstruction', 'side_input'):
                            train_module = 'jepa.four_image.train_steering'
                            train.update(condition='reconstruction' if study == 'reconstruction' else 'side',
                                         reconstruction_weight=beta, reconstruction_positions=[1],
                                         reconstruction_target_size=4, reconstruction_decoder_seed=900000+seed)
                    probe.update(seed=seed, run_id=tag+'_probe', device=device, data_root=str(data),
                                 jepa_checkpoint=train['output_checkpoint'],
                                 output_checkpoint=str(directory/'probes.pt'), output_json=str(directory/'probes.json'))
                    stages = [(train_module, train), (probe_module, probe)]
                    if study == 'traffic':
                        intervention = template('traffic_intervention')
                        intervention.update(seed=seed, run_id=tag+'_intervention', device=device,
                                            data_root=str(data), jepa_checkpoint=train['output_checkpoint'],
                                            output_json=str(directory/'intervention.json'))
                        stages.append(('jepa.traffic.intervention', intervention))
                    records.append(dict(tag=tag, study=study, weight=weight, x=x, beta=beta, seed=seed, stages=stages))
    return records


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def save_plan(records, output):
    path = output/'plan.json'
    plan = [{k:v for k,v in r.items() if k != 'stages'} for r in records]
    previous = json.loads(path.read_text()) if path.exists() else []
    assert len({r['seed'] for r in previous+plan}) == 1, 'Use a separate output directory for a different seed.'
    merged = {r['tag']:r for r in previous}
    for row in plan:
        if row['tag'] in merged:
            assert merged[row['tag']] == row
        merged[row['tag']] = row
    for record in records:
        for index, (_, config) in enumerate(record['stages']):
            saved = output/record['tag']/f'config_{index}.json'
            if saved.exists():
                assert json.loads(saved.read_text()) == config, f'Configuration changed: {saved}'
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    write_new(temporary, list(merged.values()))
    temporary.replace(path)


def execute(records, output):
    from jepa.four_image.materialize_corpus import materialize
    from jepa.four_image.feed import DictionaryFeedConfig
    from jepa.four_image.corpus import MaterializedSequenceCorpus
    from jepa.common import configure_cpu_threads
    configure_cpu_threads(8, 1)
    for number, record in enumerate(records, 1):
        print(f'[{number}/{len(records)}] {record["tag"]}', flush=True)
        train = record['stages'][0][1]
        if 'training_corpus' in train and not (output/record['tag']/'complete_0.json').exists():
            corpus = Path(train['training_corpus'])
            if not corpus.exists():
                materialize(data_root=train['data_root'], output_corpus=corpus,
                            output_json=corpus.with_suffix('.json'), feed_config=DictionaryFeedConfig(**train['feed']),
                            corpus_seed=train['train_sequence_seed'], epoch_count=25, sequences_per_epoch=100000)
            else:
                saved = MaterializedSequenceCorpus.load(corpus)
                assert saved.feed_config == DictionaryFeedConfig(**train['feed'])
                assert saved.corpus_seed == train['train_sequence_seed']
                assert saved.epoch_count == 25 and saved.sequences_per_epoch == 100000
        for index, (module, config) in enumerate(record['stages']):
            destination = output/record['tag']
            config_path = destination/f'config_{index}.json'
            completion = destination/f'complete_{index}.json'
            if completion.exists():
                assert json.loads(config_path.read_text()) == config
                from jepa.artifacts import verify_completion
                verify_completion(completion, config)
                continue
            if config_path.exists():
                raise FileExistsError(f'Incomplete stage: {config_path}')
            write_new(config_path, config)
            print(f'  Stage {index+1}: {module}; log: {destination/f"log_{index}.txt"}', flush=True)
            with (destination/f'log_{index}.txt').open('x') as log:
                subprocess.run([sys.executable, '-m', module, '--config', str(config_path)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            from jepa.artifacts import record_completion
            record_completion(completion, config)
            print(f'Completed {record["tag"]}, stage {index+1}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run the experiments and generate plots.')
    parser.add_argument('--study', choices=('all', *STUDIES), default='all')
    parser.add_argument('--main-only', action='store_true', help='Run only r=0.10 (39 encoders instead of 117).')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/cifar-10-batches-py')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--plot-only', action='store_true', help='Redraw figures from saved JSON results without training.')
    args = parser.parse_args()
    assert 0 <= args.seed < 2**63-900000
    studies = set(STUDIES) if args.study == 'all' else {args.study}
    if studies & {'reconstruction', 'side_input'}:
        studies.add('predictability')
    weights = (.10,) if args.main_only else (.05, .10, .20)
    output = args.output.expanduser().resolve()
    if args.plot_only:
        from plot import plot
        plot(output)
        print(f'Figures: {output/"figures/index.html"}')
        return
    records = jobs(studies, weights, args.seed, args.data_root.expanduser().resolve(), output, args.device)
    for record in records:
        module, config = record['stages'][0]
        importlib.import_module(module).Config.from_mapping(config)
    if args.plan:
        print(json.dumps([{k:v for k,v in record.items() if k != 'stages'} for record in records], indent=2))
        return
    save_plan(records, output)
    print(f'{len(records)} encoder settings; completed stages will be verified and reused.', flush=True)
    from jepa.common import resolve_device
    resolve_device(args.device, bf16=True)
    from download_data import download, verify
    data = args.data_root.expanduser().resolve()
    if data == ROOT/'data/cifar-10-batches-py':
        download(data.parent)
    else:
        verify(data)
    execute(records, output)
    from plot import plot
    plot(output)
    print(f'Figures: {output/"figures/index.html"}')


if __name__ == '__main__':
    main()
