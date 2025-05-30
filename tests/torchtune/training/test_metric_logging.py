# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import tempfile
from io import StringIO
from typing import cast
from unittest.mock import patch

import mlflow

import mlflow.artifacts
import pytest
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_utils import assert_expected, captured_output

from torchtune.training.metric_logging import (
    CometLogger,
    DiskLogger,
    MLFlowLogger,
    StdoutLogger,
    TensorBoardLogger,
    WandBLogger,
)


class TestDiskLogger:
    def test_log_txt(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                assert values[i] == f"Step {i} | test_log:{float(i) ** 2}\n"

    def test_log_dict_txt(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir)
            for i in range(5):
                logger.log_dict(step=i, payload={"metric_1": i, "metric_2": i**2})
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                assert values[i] == f"Step {i} | metric_1:{i} metric_2:{i**2} \n"

    def test_log_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir, output_fmt="jsonl")
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                json_line = json.loads(values[i].strip())
                assert json_line == {
                    "step": i,
                    "test_log": float(i) ** 2,
                }, f"Unexpected JSON line: {json_line}"

    def test_log_dict_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir, output_fmt="jsonl")
            for i in range(5):
                logger.log_dict(step=i, payload={"metric_1": i, "metric_2": i**2})
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                json_line = json.loads(values[i].strip())
                assert json_line == {
                    "step": i,
                    "metric_1": i,
                    "metric_2": i**2,
                }, f"Unexpected JSON line: {json_line}"


class TestStdoutLogger:
    def test_stdout_log(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log(step=0, name="metric_1", data=1.1)
            out = cast(StringIO, out)
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\n"
            ), f"Actual output: {out.getvalue()}"

            logger.log(step=1, name="metric_1", data=2.1)
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n"
            ), f"Actual output: {out.getvalue()}"

            logger.close()
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n"
            ), f"Actual output: {out.getvalue()}"

    def test_stdout_log_dict(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log_dict(step=0, payload={"metric_1": 1, "metric_2": 1})
            out = cast(StringIO, out)
            assert (
                out.getvalue() == "Step 0 | metric_1:1 metric_2:1 \n"
            ), f"Actual output: {out.getvalue()}"

            logger.log_dict(
                step=1, payload={"metric_1": 2, "metric_2": 2.2, "metric_3": 2.2344}
            )
            assert (
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n"
            ), f"Actual output: {out.getvalue()}"

            logger.close()
            assert (
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n"
            ), f"Actual output: {out.getvalue()}"


class TestTensorBoardLogger:
    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            acc = EventAccumulator(logger.log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                assert_expected(event.tensor_proto.float_val[0], float(i) ** 2)
                assert_expected(event.step, i)

    def test_log_dict(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            acc = EventAccumulator(logger.log_dir)
            acc.Reload()
            for i in range(5):
                tensor_tag = acc.Tensors(f"log_dict_{i}")[0]
                assert_expected(tensor_tag.tensor_proto.float_val[0], float(i) ** 2)
                assert_expected(tensor_tag.step, 1)


@pytest.mark.skip(reason="This was never running and needs to be fixed")
class TestWandBLogger:
    def test_log(self) -> None:
        with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
            logger = WandBLogger(project="test_project")
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            assert mock_log.call_count == 5
            for i in range(5):
                mock_log.assert_any_call({"test_log": float(i) ** 2}, step=i)

    def test_log_dict(self) -> None:
        with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
            logger = WandBLogger(project="test_project")
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            mock_log.assert_called_with(metric_dict, step=1)

    def test_save_config(self) -> None:
        with (
            patch("wandb.init") as mock_init,
            patch("wandb.run", create=True) as mock_run,
            patch("OmegaConf.save") as mock_save,
            patch("wandb.save") as mock_wandb_save,
        ):
            logger = WandBLogger(project="test_project")
            cfg = OmegaConf.create({"a": 1, "b": 2})

            with patch.object(logger, "_wandb", mock_run):
                logger.save_config(cfg)

            expected_config_path = "torchtune_config.yaml"
            mock_save.assert_called_once_with(cfg, expected_config_path)
            mock_wandb_save.assert_called_once_with(expected_config_path)


class TestCometLogger:
    def test_log(self) -> None:
        with patch("comet_ml.start") as mock_experiment:
            logger = CometLogger(project="test_project")
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            assert mock_experiment.return_value.log_metric.call_count == 5
            for i in range(5):
                mock_experiment.return_value.log_metric.assert_any_call(
                    "test_log", float(i) ** 2, step=i
                )

    def test_log_dict(self) -> None:
        with patch("comet_ml.start") as mock_experiment:
            logger = CometLogger(project="test_project")
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            mock_experiment.return_value.log_metrics.assert_called_with(
                metric_dict, step=1
            )

    def test_log_config(self) -> None:
        with patch("comet_ml.start") as mock_experiment:
            logger = CometLogger(project="test_project")
            cfg = OmegaConf.create({"a": 1, "b": 2})
            logger.log_config(cfg)
            mock_experiment.return_value.log_parameters.assert_called_with(cfg)


@pytest.fixture(scope="class")
def mlflow_context_fixture():
    original_uri = mlflow.get_tracking_uri()

    with tempfile.TemporaryDirectory() as tmpdir:
        mlflow.set_tracking_uri(f"file:{tmpdir}")
        yield

    # Restore the original URI
    mlflow.set_tracking_uri(original_uri)


@pytest.mark.usefixtures("mlflow_context_fixture")
class TestMLFlowLogger:
    def test_log(self):
        logger = MLFlowLogger(experiment_name="my_experiment", run_name="run1")
        run_id = logger._run_id
        logger.log("test_metric", 1.0, step=1)
        logger.close()

        run = mlflow.get_run(run_id)
        assert run.data.metrics == {"test_metric": 1}

    def test_log_dict(self):
        logger = MLFlowLogger(experiment_name="my_experiment", run_name="run2")
        run_id = logger._run_id
        metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
        logger.log_dict(metric_dict, step=2)
        logger.close()

        run = mlflow.get_run(run_id)
        assert run.data.metrics == metric_dict

    def test_log_config(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = OmegaConf.create(
                {"foo": {"bar": "baz"}, "qux": "quux", "output_dir": output_dir}
            )
            logger = MLFlowLogger(experiment_name="my_experiment", run_name="run2")
            run_id = logger._run_id

            logger.log_config(cfg)

        expected = {"foo.bar": "baz", "qux": "quux", "output_dir": output_dir}

        run = mlflow.get_run(run_id)
        assert run.data.params == expected

        artifacts = mlflow.artifacts.list_artifacts(
            run_id=run_id, artifact_path=output_dir.lstrip("/")
        )
        assert len(artifacts) == 1
        assert artifacts[0].path.endswith("torchtune_config.yaml")
