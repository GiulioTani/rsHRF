from typing import Any

import sys
import pytest
from unittest import mock
from .. import CLI
import numpy as np
import nibabel as nib
from ..utils.default_parameters import default_parameters

SHAPE = (10, 10, 10, 10)
mockTR = 2


def get_data(image_type):
    data = np.zeros(SHAPE, dtype=np.int16)

    if image_type == "nifti":
        data = nib.Nifti1Image(data, np.eye(4))
        hdr = data.header
        hdr.set_zooms(mockTR * np.ones(len(SHAPE)))
        data = nib.Nifti1Image(data, np.eye(4), hdr)
    else:
        data = nib.gifti.GiftiDataArray(data.astype(np.float32), datatype="float32")
    return data


def test_GUI(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rsHRF", "GUI", "--no-bids"])
    with mock.patch("rsHRF.rsHRF_GUI.run.run") as mock_call:
        CLI.run_rsHRF()
        mock_call.assert_called_once()


def test_text(monkeypatch, tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("mock", encoding="utf-8")
    with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
        monkeypatch.setattr(
            sys, "argv", ["rsHRF", p._str, d._str, "--no-bids", "--TR", "2"]
        )
        CLI.run_rsHRF()
        mock_call.assert_called_once()

    monkeypatch.setattr(sys, "argv", ["rsHRF", p._str, d._str, "--TR", "2"])
    with pytest.raises(SystemExit):
        CLI.run_rsHRF()

    monkeypatch.setattr(sys, "argv", ["rsHRF", p._str, d._str, "--no-bids"])
    with pytest.raises(SystemExit):
        CLI.run_rsHRF()

    with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
        with pytest.warns(Warning):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "rsHRF",
                    p._str,
                    d._str,
                    "--no-bids",
                    "--TR",
                    "2",
                    "--participant-label",
                    "001",
                ],
            )
            CLI.run_rsHRF()
        mock_call.assert_called_once()


def test_temporal_mask(monkeypatch, tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("mock", encoding="utf-8")
    m = d / "mask.txt"
    m.write_text("01011100", encoding="utf-8")
    with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsHRF",
                p._str,
                d._str,
                "--no-bids",
                "--TR",
                "2",
                "--temporal-mask",
                m._str,
            ],
        )
        CLI.run_rsHRF()
        mock_call.assert_called_once()

    n = d / "mask2.txt"
    n.write_bytes(np.arange(5, dtype=float).tobytes())
    with pytest.raises(SystemExit):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rsHRF",
                p._str,
                d._str,
                "--no-bids",
                "--TR",
                "2",
                "--temporal-mask",
                n._str,
            ],
        )
        CLI.run_rsHRF()


def test_bad_paths(monkeypatch, tmp_path):
    di = tmp_path / "inp"
    di.mkdir()
    p = di / "hello.txt"
    p.write_text("mock", encoding="utf-8")
    do = tmp_path / "outp"
    do.mkdir()

    cli_inputs = [
        ["rsHRF", di._str, do._str],  # assumes BIDS input but "participant" is missing
        [
            "rsHRF",
            p._str,
            "--no-bids",
            "--TR",
            "2",
        ],  # assumes text but no output directory
        [
            "rsHRF",
            p._str,
            str(tmp_path / "fake"),
            "--no-bids",
            "--TR",
            "2",
        ],  # assumes text but missing output directory
        [
            "rsHRF",
            str(di / "fake.txt"),
            do._str,
            "--no-bids",
            "--TR",
            "2",
        ],  # assumes text but missing input file
    ]
    for cli_input in cli_inputs:
        monkeypatch.setattr(sys, "argv", cli_input)
        with pytest.raises(SystemExit):
            CLI.run_rsHRF()


def test_gnii(monkeypatch, tmp_path):
    test_file_1 = "test.gii"
    test_file_2 = "test.gii.gz"
    test_file_3 = "test.nii"
    test_file_4 = "test.nii.gz"
    test_files = [test_file_1, test_file_2, test_file_3, test_file_4]
    di = tmp_path / "inp"
    di.mkdir()
    with mock.patch("nibabel.load") as load_mock:
        for test_file in test_files:
            if "nii" in test_file:
                load_mock.return_value = get_data("nifti")
            elif "gii" in test_file:
                load_mock.return_value = get_data("gifti")
            p = di / test_file
            p.write_text("mock", encoding="utf-8")

            monkeypatch.setattr(
                sys,
                "argv",
                ["rsHRF", p._str, di._str, "--no-bids"]
                + (["--TR", "2"] if "gii" in test_file else []),
            )
            with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
                CLI.run_rsHRF()
                para = {
                    k: default_parameters[k]
                    for k in [
                        "estimation",
                        "passband",
                        "passband_deconvolve",
                        "T",
                        "T0",
                        "TD_DD",
                        "AR_lag",
                        "thr",
                        "order",
                        "len",
                        "min_onset_search",
                        "max_onset_search",
                        "localK",
                        "wiener",
                    ]
                }
                para["temporal_mask"] = None
                para["TR"] = mockTR
                para["dt"] = para["TR"] / para["T"]
                para["lag"] = np.arange(
                    np.trunc(para["min_onset_search"] / para["dt"]),
                    np.trunc(para["max_onset_search"] / para["dt"]) + 1,
                    dtype="int",
                )
                mock_call.assert_called_once()

                call_args = mock_call.call_args
                print("call_args[0]", call_args[0])
                for good, mocked in zip(
                    [
                        p._str,
                        None,
                        di._str,
                        para,
                        -1,
                        test_file[4:],
                        "input",
                        [],
                        False,
                    ],
                    call_args[0],
                ):
                    print("good", good)
                    print("mocked", mocked)
                    if isinstance(good, dict):
                        good = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in good.items()
                        }
                        mocked = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in mocked.items()
                        }
                        assert good == mocked
                    else:
                        assert good == mocked


def test_bad_masks(monkeypatch, tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("mock", encoding="utf-8")
    m = d / "mask.bad"
    m.write_text("bad", encoding="utf-8")
    with pytest.warns(Warning):
        with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
            monkeypatch.setattr(
                sys,
                "argv",
                ["rsHRF", p._str, d._str, "--no-bids", "--TR", "2", "-m", m._str],
            )
            CLI.run_rsHRF()
            mock_call.assert_called_once()

    with mock.patch("nibabel.load") as load_mock:
        load_mock.return_value = get_data("nifti")
        p = d / "test.nii"
        p.write_text("mock", encoding="utf-8")

        with mock.patch("rsHRF.fourD_rsHRF.demo_rsHRF") as mock_call:
            monkeypatch.setattr(
                sys, "argv", ["rsHRF", p._str, d._str, "--no-bids", "-m", "BIDS"]
            )
            with pytest.warns(Warning):
                CLI.run_rsHRF()
            mock_call.assert_called_once()

        with pytest.raises(SystemExit):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "rsHRF",
                    p._str,
                    d._str,
                    "--no-bids",
                    "-m",
                    str(d / "fake.gii"),
                ],  # format mismatch
            )
            CLI.run_rsHRF()

        with pytest.raises(SystemExit):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "rsHRF",
                    p._str,
                    d._str,
                    "--no-bids",
                    "-m",
                    str(d / "fake.nii"),
                ],  # missing file
            )
            CLI.run_rsHRF()

        with pytest.raises(SystemExit):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "rsHRF",
                    p._str,
                    d._str,
                    "--no-bids",
                    "-m",
                    m._str,
                ],  # wrong extension
            )
            CLI.run_rsHRF()
