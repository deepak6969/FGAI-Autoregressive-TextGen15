"""
Foundations of Generative AI (ICT4403) - Assignment 2
Autoregressive Text Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU

This script implements word-level autoregressive language modeling on the WikiText-2 dataset.
It provides training, evaluation, checkpoint restoration, and temperature-controlled sequential
text generation with diversity metrics calculation.
"""

import argparse
import copy
import json
import math
import os
import platform
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Set deterministic cuBLAS workspace config
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

MODELS = ['SimpleRNN', 'BiRNN', 'LSTM', 'GRU']
PROMPTS = [
    'the history of the united states began with',
    'during the second world war the army was',
    'the album received positive reviews from music critics'
]

DEFAULT_CONFIG = {
    'dataset': 'Salesforce/wikitext',
    'dataset_config': 'wikitext-2-raw-v1',
    'train_entries': 8000,
    'validation_entries': 1500,
    'min_chars': 20,
    'vocab_limit': 10000,
    'seq_len': 30,
    'stride': 3,
    'embed_dim': 256,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 128,
    'epochs': 8,
    'lr': 0.001,
    'clip': 1.0,
    'training_seeds': [42],
    'sampling_seeds': [101, 202, 303],
    'temperatures': [0.5, 1.0, 1.3],
    'new_tokens': 25,
    'prompts': PROMPTS
}


def write_json(path, value):
    """Write data to a JSON file formatted with indentation."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tokenize(text):
    """Lowercase text and extract words and standard punctuation as individual tokens."""
    return re.findall(r'\b\w+\b|[.,!?;]', text.lower())


def seed_all(seed):
    """Ensure strict reproducibility across Python, NumPy, and PyTorch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Windows(Dataset):
    """Sliding-window dataset constructing (input_sequence, next_token_target) pairs."""
    def __init__(self, ids, seq_len, stride=3):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.starts = range(0, max(0, len(ids) - seq_len), stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = self.starts[i]
        return self.ids[start:start + self.seq_len], self.ids[start + self.seq_len]


def prepare_data(raw, config):
    """
    Preprocess raw dataset entries, build training vocabulary, and construct sliding windows.
    Splits are kept strictly isolated.
    """
    tokens, stats = {}, {}
    for split in ['train', 'validation', 'test']:
        entries = list(raw[split]['text'])
        eligible = entries if split == 'test' else [s for s in entries if len(s.strip()) > config['min_chars']]
        cap = config.get(split + '_entries') if split != 'test' else None
        selected = eligible[:cap] if cap is not None else eligible
        tokens[split] = [w for s in selected for w in tokenize(s)]
        stats[split] = dict(
            raw_entries=len(entries),
            eligible_entries=len(eligible),
            selected_entries=len(selected),
            tokens=len(tokens[split])
        )

    # Vocabulary built strictly from training split
    itos = ['<pad>', '<unk>'] + [w for w, _ in Counter(tokens['train']).most_common(config['vocab_limit'] - 2)]
    stoi = {w: i for i, w in enumerate(itos)}

    datasets = {}
    for split, words in tokens.items():
        ids = [stoi.get(w, 1) for w in words]
        datasets[split] = Windows(ids, config['seq_len'], config['stride'])
        stats[split].update(
            sequences=len(datasets[split]),
            unknown_tokens=ids.count(1),
            unknown_rate=ids.count(1) / max(1, len(ids))
        )
        if not len(datasets[split]):
            raise ValueError(f'No windows created in split: {split}')

    stats.update(
        vocabulary_size=len(itos),
        dataset=config['dataset'],
        dataset_config=config['dataset_config'],
        fingerprints={s: getattr(raw[s], '_fingerprint', None) for s in tokens}
    )
    return datasets, itos, stats


class LanguageModel(nn.Module):
    """
    Unified Language Model supporting SimpleRNN, BiRNN, LSTM, and GRU backbones.
    """
    def __init__(self, kind, vocab_size, config):
        super().__init__()
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, config['embed_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        recurrent_map = {
            'SimpleRNN': nn.RNN,
            'BiRNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        recurrent_cls = recurrent_map[kind]
        self.rnn = recurrent_cls(
            config['embed_dim'],
            config['hidden_dim'],
            num_layers=config['num_layers'],
            batch_first=True,
            bidirectional=(kind == 'BiRNN'),
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )
        fc_in = config['hidden_dim'] * (2 if kind == 'BiRNN' else 1)
        self.fc = nn.Linear(fc_in, vocab_size)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, hidden = self.rnn(emb)
        if self.kind == 'LSTM':
            hidden = hidden[0]  # extract hidden state from (h, c) tuple
        if self.kind == 'BiRNN':
            # Concatenate final forward and backward states of the top layer
            final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final = hidden[-1]
        return self.fc(self.dropout(final))


def loader(dataset, config, seed, training=False):
    """Create DataLoader with deterministic generator."""
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=training,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )


def synchronize(device):
    """Synchronize CUDA operations for precise runtime profiling."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def pass_epoch(model, batches, device, optimizer=None, clip=1.0):
    """Execute a single training or evaluation epoch."""
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
            count += y.numel()

    loss = total_loss / count
    if not math.isfinite(loss):
        raise RuntimeError('Encountered non-finite loss during training/evaluation')
    return dict(
        loss=loss,
        perplexity=math.exp(loss),
        accuracy_pct=100.0 * correct / count,
        targets=count
    )


def restore(path, kind, itos, config, device):
    """Restore model from checkpoint state dict."""
    model = LanguageModel(kind, len(itos), config).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def fit(kind, datasets, itos, config, seed, directory, device):
    """
    Train a model for specified epochs, checkpoint on best validation loss,
    and evaluate final test set metrics.
    """
    seed_all(seed)
    model = LanguageModel(kind, len(itos), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    train = loader(datasets['train'], config, seed, True)
    val = loader(datasets['validation'], config, seed)
    checkpoint = directory / f'{kind}-seed{seed}-best.pt'
    history, best_loss, best_epoch = [], float('inf'), None

    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, config['epochs'] + 1):
        synchronize(device)
        train_start = time.perf_counter()
        tr = pass_epoch(model, train, device, optimizer, config['clip'])
        synchronize(device)
        train_seconds = time.perf_counter() - train_start

        va = pass_epoch(model, val, device)
        if va['loss'] < best_loss:
            best_loss, best_epoch = va['loss'], epoch
            torch.save(model.state_dict(), checkpoint)

        row = dict(
            model=kind,
            training_seed=seed,
            epoch=epoch,
            train_seconds=train_seconds,
            **{'train_' + k: v for k, v in tr.items()},
            **{'validation_' + k: v for k, v in va.items()}
        )
        history.append(row)
        write_json(directory / f'{kind}-seed{seed}-history.json', history)
        print(f"[{kind}] Seed={seed} Epoch={epoch} | Train Loss={tr['loss']:.4f} PPL={tr['perplexity']:.2f} | Val Loss={va['loss']:.4f} PPL={va['perplexity']:.2f} Acc={va['accuracy_pct']:.2f}%")

    synchronize(device)
    elapsed = time.perf_counter() - start
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    del optimizer, model
    best = restore(checkpoint, kind, itos, config, device)
    best_val = pass_epoch(best, val, device)
    test = pass_epoch(best, loader(datasets['test'], config, seed), device)

    summary = dict(
        model=kind,
        training_seed=seed,
        parameters=parameters,
        best_epoch=best_epoch,
        checkpoint=checkpoint.name,
        training_seconds=sum(r['train_seconds'] for r in history),
        fit_with_validation_checkpoint_history_seconds=elapsed,
        final_train_loss=tr['loss'],
        **{'final_validation_' + k: v for k, v in va.items()},
        **{'best_validation_' + k: v for k, v in best_val.items()},
        **{'test_' + k: v for k, v in test.items()}
    )
    return best, history, summary


@torch.no_grad()
def generate(model, prompt, itos, config, temperature, sampling_seed):
    """
    Autoregressively generate next tokens given a seed prompt using temperature scaling.
    Special tokens (<pad>, <unk>) are masked out to guarantee informative generation.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be a finite positive number')
    words = tokenize(prompt)
    if not words:
        raise ValueError('Prompt must contain at least one valid token')

    stoi = {w: i for i, w in enumerate(itos)}
    context = [stoi.get(w, 1) for w in words][-config['seq_len']:]
    device = next(model.parameters()).device

    rng = torch.Generator(device='cpu').manual_seed(sampling_seed)
    continuation = []
    model.eval()

    for _ in range(config['new_tokens']):
        logits = model(torch.tensor([context], device=device))[0].float().cpu()
        logits[0:2] = -float('inf')  # Mask <pad> and <unk>
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, 1, generator=rng).item()
        continuation.append(itos[token])
        context = (context + [token])[-config['seq_len']:]
    return continuation


def diversity(tokens):
    """Compute Distinct-1 (TTR), Distinct-2, and Repeated-Bigram Fraction."""
    bigrams = list(zip(tokens, tokens[1:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return dict(
        distinct_1=len(set(tokens)) / len(tokens) if tokens else 0.0,
        distinct_2=d2,
        repeated_bigram_fraction=1.0 - d2 if bigrams else 0.0
    )


def sample_model(model, kind, itos, config, summary):
    """Generate continuations across all prompts, temperatures, and sampling seeds."""
    rows = []
    for p, prompt in enumerate(config['prompts'], 1):
        for temperature in config['temperatures']:
            for seed in config['sampling_seeds']:
                continuation = generate(model, prompt, itos, config, temperature, seed)
                rows.append(dict(
                    sample_id=f'{kind}-train{summary["training_seed"]}-p{p}-t{temperature}-s{seed}',
                    model=kind,
                    training_seed=summary['training_seed'],
                    checkpoint=summary['checkpoint'],
                    sampling_seed=seed,
                    temperature=temperature,
                    prompt=prompt,
                    continuation=' '.join(continuation),
                    continuation_tokens=len(continuation),
                    **diversity(continuation)
                ))
    return rows


def export(directory):
    """Produce summary tables, learning curve plots, and temperature metrics."""
    directory = Path(directory)
    results = pd.read_csv(directory / 'results.csv')
    history = pd.read_csv(directory / 'history.csv')
    samples = pd.read_csv(directory / 'samples.csv')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))

    metrics = ['distinct_1', 'distinct_2', 'repeated_bigram_fraction']
    agg = samples.groupby(['model', 'temperature'])[metrics].agg(['mean', 'std', 'count']).reset_index()
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    agg.to_csv(directory / 'temperature_summary.csv', index=False)

    # Learning Curves Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for (model, seed), frame in history.groupby(['model', 'training_seed']):
        for ax, metric in zip(axes, ['loss', 'perplexity', 'accuracy_pct']):
            ax.plot(frame.epoch, frame['validation_' + metric], label=f'{model}')
            ax.set(xlabel='Epoch', ylabel='Validation ' + metric.replace('_pct', ' (%)'))
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
    axes[0].set_title('Validation Loss')
    axes[1].set_title('Validation Perplexity')
    axes[2].set_title('Validation Accuracy (%)')
    fig.tight_layout()
    fig.savefig(directory / 'learning_curves.png', dpi=160)
    plt.close(fig)

    # Temperature Diversity Metrics Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metric_labels = ['Distinct-1 (TTR)', 'Distinct-2', 'Repeated Bigram Fraction']
    for model, frame in agg.groupby('model'):
        for ax, metric in zip(axes, metrics):
            ax.errorbar(
                frame.temperature,
                frame[metric + '_mean'],
                yerr=frame[metric + '_std'].fillna(0),
                marker='o',
                capsize=3,
                label=model
            )
            ax.set(xlabel='Sampling Temperature', ylim=(-0.05, 1.05))
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
    for ax, label in zip(axes, metric_labels):
        ax.set_title(label)
    fig.suptitle('Sampling Temperature vs Generation Diversity Metrics', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(directory / 'temperature_metrics.png', dpi=160)
    plt.close(fig)
    print(f"Export completed. Plots and summaries updated in {directory}")


def run(config=None, output='.'):
    """Run full pipeline: data loading, model training, evaluation, sampling, and export."""
    from datasets import load_dataset
    import datasets as hf_datasets

    config = dict(DEFAULT_CONFIG if config is None else config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write_json(output / 'environment.json', dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        matplotlib=matplotlib.__version__,
        datasets=hf_datasets.__version__,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        deterministic_algorithms=True,
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    ))

    print(f"Loading dataset: {config['dataset']} ({config['dataset_config']})...")
    raw = load_dataset(config['dataset'], config['dataset_config'])
    datasets, itos, stats = prepare_data(raw, config)
    write_json(output / 'dataset_statistics.json', stats)
    write_json(output / 'vocabulary.json', itos)
    print(f"Vocabulary size: {len(itos)} tokens")
    print(f"Train sequences: {stats['train']['sequences']}, Val sequences: {stats['validation']['sequences']}, Test sequences: {stats['test']['sequences']}")

    histories, results, samples = [], [], []
    for seed in config['training_seeds']:
        for kind in MODELS:
            print(f"\n==================== Training {kind} (Seed {seed}) ====================")
            model, history, summary = fit(kind, datasets, itos, config, seed, output, device)
            histories.extend(history)
            results.append(summary)
            print(f"Generating samples for {kind}...")
            samples.extend(sample_model(model, kind, itos, config, summary))

            pd.DataFrame(histories).to_csv(output / 'history.csv', index=False)
            pd.DataFrame(results).to_csv(output / 'results.csv', index=False)
            pd.DataFrame(samples).to_csv(output / 'samples.csv', index=False)
            del model

    write_json(output / 'history.json', histories)
    export(output)
    return pd.DataFrame(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoregressive Recurrent Language Modeling')
    parser.add_argument('--action', choices=['train', 'export', 'sample'], default='export',
                        help='Action to perform: train full models, export existing results, or sample')
    parser.add_argument('--output', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    if args.action == 'train':
        run(DEFAULT_CONFIG, args.output)
    elif args.action == 'export':
        export(args.output)
