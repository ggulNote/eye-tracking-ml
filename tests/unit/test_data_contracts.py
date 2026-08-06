from pathlib import Path

from ggulnote_ml.data import build_data_source, split_by_participant


def test_synthetic_data_and_participant_split(pipeline_config, tmp_path: Path) -> None:
    dataset = build_data_source(
        pipeline_config.data,
        pipeline_config.training.seed,
        tmp_path,
    ).load()
    dataset.validate(pipeline_config.data.sequence_length)
    splits = split_by_participant(
        dataset,
        pipeline_config.data,
        pipeline_config.training.seed,
    )
    train_ids = {sample.participant_id for sample in splits.train.samples}
    validation_ids = {sample.participant_id for sample in splits.validation.samples}
    test_ids = {sample.participant_id for sample in splits.test.samples}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

