from torch.utils.data import DataLoader

from wesep.dataset.dataset import Dataset
from wesep.dataset.dataset import tse_collate_fn
from wesep.utils.file_utils import load_speaker_embeddings


def test_premixed_dataset():
    configs = {
        "shuffle": False,
        "shuffle_args": {"shuffle_size": 2500},
        "resample_rate": 16000,
        "chunk_len": 32000,
    }

    spk2embed_dict = load_speaker_embeddings(
        "data/clean/test/embed.scp", "data/clean/test/single.utt2spk"
    )

    dataset = Dataset(
        "shard",
        "data/clean/test/shard.list",
        configs=configs,
        spk2embed_dict=spk2embed_dict,
        whole_utt=False,
    )
    return dataset


def test_online_dataset():
    # Implementation to test the online speaker mixing dataloader
    configs = {
        "shuffle": True,
        "resample_rate": 16000,
        "chunk_len": 64000,
        "num_speakers": 2,
        "online_mix": True,
        "reverb": False,
    }

    spk2embed_dict = load_speaker_embeddings(
        "mydata/clean/test/embed.scp", "mydata/clean/test/utt2spk"
    )
    dataset = Dataset(
        "shard",
        "mydata/clean/test/shard.list",
        configs=configs,
        spk2embed_dict=spk2embed_dict,
        whole_utt=False,
    )

    return dataset


if __name__ == "__main__":
    dataset = test_online_dataset()

    dataloader = DataLoader(
        dataset, batch_size=4, num_workers=1, collate_fn=tse_collate_fn
    )

    for i, batch in enumerate(dataloader):
        print(
            batch["wav_mix"].size(),
            batch["wav_targets"].size(),
            batch["spk_embeds"].size(),
        )
        if i == 0:
            break
