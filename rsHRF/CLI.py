import sys
import numpy as np
import os.path as op
import json
from argparse import ArgumentParser
from bids.layout import BIDSLayout
from bids.config import set_option
from pathlib import Path
from rsHRF import spm_dep, fourD_rsHRF, utils


import warnings
from .utils.default_parameters import default_parameters, available_estimations

set_option("extension_initial_dot", True)

with open(op.join(op.dirname(op.realpath(__file__)), "VERSION"), "r") as fh:
    __version__ = fh.read().strip("\n")


def get_parser():
    parser = ArgumentParser(
        description="retrieves the onsets of pseudo-events triggering a "
        "haemodynamic response from resting state fMRI BOLD "
        "voxel-wise signal"
    )

    parser.add_argument(
        "bids_dir",
        help="the input data for the analysis: a path to a data file or the root "
        "folder of a BIDS valid dataset, or 'GUI' to run in graphical user interface "
        "mode",
        default="GUI",
    )

    parser.add_argument(
        "output_dir",
        action="store",
        type=op.abspath,
        help="the output path for the outcomes of processing",
        nargs="?",
    )

    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. "
        "Multiple participant level analyses can be run independently "
        "(in parallel) using the same output_dir. Only 'participant' level analysis is"
        " allowed.",
        choices=["participant"],
        nargs="?",
    )

    parser.add_argument(
        "--no-bids",
        action="store_true",
        help="Explicitly disable bids input. Necessary when using txt or nii/gii files"
        " directly.",
    )

    parser.add_argument(
        "--n_jobs",
        action="store",
        type=int,
        default=-1,
        help="the number of parallel processing elements to use, default is -1 (use "
        "all available cores)",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="rsHRF version {}".format(__version__),
    )

    parser.add_argument(
        "--participant-label",
        help="The label(s) of the participant(s) that should be analyzed. The label "
        "corresponds to sub-<participant_label> from the BIDS spec "
        '(so it does not include "sub-"). If this parameter is not '
        "provided all subjects should be analyzed. Multiple "
        "participants can be specified with a space separated list.",
        nargs="+",
    )

    parser.add_argument(
        "--bids-filter-file",
        action="store",
        type=op.abspath,
        help="a JSON file describing custom BIDS input filters using PyBIDS. "
        "For further details, please check out http://bids-apps.neuroimaging.io/rsHRF/",
    )

    parser.add_argument(
        "-m",
        "--mask",
        action="store",
        type=str,
        help="the absolute path to a single mask file, which should be of the same "
        "type as the input file (NIfTI or GIfTI). Use 'BIDS' to enable the use of "
        "mask files present in the BIDS directory itself.",
    )

    group_para = parser.add_argument_group("Parameters")

    group_para.add_argument(
        "--estimation",
        action="store",
        choices=available_estimations,
        help="Choose the estimation procedure from "
        "canon2dd (canonical shape with 2 derivatives), "
        "sFIR (smoothed Finite Impulse Response), "
        "FIR (Finite Impulse Response), "
        "fourier (Fourier Basis Set), "
        "hanning (Fourier Basis w Hanning), "
        f"gamma (Gamma Basis Set).",
        default=default_parameters["estimation"],
    )

    group_para.add_argument(
        "--passband",
        action="store",
        type=float,
        nargs=2,
        metavar=("LOW_FREQ", "HIGH_FREQ"),
        default=default_parameters["passband"],
        help="set intervals for bandpass filter, default is 0.01 - 0.08",
    )

    group_para.add_argument(
        "--passband-deconvolve",
        action="store",
        type=float,
        nargs=2,
        metavar=("LOW_FREQ", "HIGH_FREQ"),
        default=default_parameters["passband_deconvolve"],
        help="set intervals for bandpass filter (used while deconvolving BOLD), "
        "default is no-filtering",
    )

    group_para.add_argument(
        "--TR",
        action="store",
        type=float,
        help="set TR parameter",
        default=default_parameters["TR"],
    )

    group_para.add_argument(
        "--T",
        "-T",
        action="store",
        type=int,
        help=f"set T parameter, default is {default_parameters['T']}",
        default=default_parameters["T"],
    )

    group_para.add_argument(
        "--T0",
        action="store",
        type=int,
        default=default_parameters["T0"],
        help=f"set T0 parameter, default is {default_parameters['T0']}",
    )

    group_para.add_argument(
        "--TD",
        action="store",
        dest="TD_DD",
        type=int,
        default=default_parameters["TD_DD"],
        help=f"set TD_DD parameter, default is {default_parameters['TD_DD']}",
    )

    group_para.add_argument(
        "--AR-lag",
        action="store",
        type=int,
        default=default_parameters["AR_lag"],
        help=f"set AR_lag parameter, default is {default_parameters['AR_lag']}",
    )

    group_para.add_argument(
        "--thr",
        "--threshold",
        action="store",
        type=float,
        default=default_parameters["thr"],
        help=f"set thr parameter, default is {default_parameters['thr']}",
    )

    group_para.add_argument(
        "--temporal-mask",
        "--tmask",
        action="store",
        type=op.abspath,
        help="the path for the (temporal) mask file.\n The mask file should be a text "
        "file, a sequence of 0s and 1s of the same length as the signal",
    )

    group_para.add_argument(
        "--order",
        action="store",
        type=int,
        default=default_parameters["order"],
        help=f"set the number of basis vectors, default is "
        f"{default_parameters['order']} (used for fourier, hanning and gamma basis "
        "functions)",
    )

    group_para.add_argument(
        "--len",
        action="store",
        type=int,
        default=default_parameters["len"],
        help=f"set len parameter, default is {default_parameters['len']}s",
    )

    group_para.add_argument(
        "--min-onset-search",
        action="store",
        type=int,
        default=default_parameters["min_onset_search"],
        help=f"set min_onset_search parameter, default is {default_parameters['min_onset_search']}s",
    )

    group_para.add_argument(
        "--max-onset-search",
        action="store",
        type=int,
        default=default_parameters["max_onset_search"],
        help=f"set max_onset_search parameter, default is {default_parameters['max_onset_search']}s",
    )

    group_para.add_argument(
        "--localK",
        action="store",
        type=int,
        help=f"set localK, default is {default_parameters['localK']}",
        default=default_parameters["localK"],
    )

    group_para.add_argument(
        "--wiener",
        action="store_true",
        help="to perform iterative wiener deconvolution",
    )

    return parser


def run_rsHRF():
    parser = get_parser()
    args = parser.parse_args()
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    para = arg_groups["Parameters"]
    temporal_mask = []

    if args.bids_dir == "GUI" and args.no_bids:
    try:
        from .rsHRF_GUI import run as gui_run
    except Exception as exc:
        parser.error(
            "--GUI could not be started. This is expected in headless or Docker "
            f"environments. Original error: {exc}"
        )

    gui_run.run(para)
    return 0
    else:
        if args.output_dir is None:
            parser.error(
                "--output_dir is required when executing in command-line interface"
            )
        elif not op.isdir(args.output_dir):
            parser.error(
                "--output_dir must be a valid directory path that already exists."
            )

        if not op.exists(args.bids_dir):
            parser.error(
                "The input path provided does not exist, please provide a valid path."
            )

        if op.isdir(args.bids_dir):
            input_type = "BIDS"
            if args.analysis_level is None:
                parser.error(
                    "When running BIDS analysis you must provide the analysis level 'participant'."
                )
        elif (
            args.bids_dir.endswith((".nii", ".nii.gz", ".gii", ".gii.gz"))
            and args.no_bids
        ):
            input_type = "4Dimage"
        elif args.bids_dir.endswith(".txt") and args.no_bids:
            input_type = "text"
        else:
            parser.error(
                "When not using BIDS structure you must specify --no-bids and the input file "
                "should be a 4D NIfTI or GIfTI file, or a text file containing the time-series"
            )

        if input_type != "BIDS" and args.participant_label is not None:
            warnings.warn(
                "Participant_labels are not to be used with 4Dimage or text input, do not supply it",
            )

        if input_type == "text" and args.mask is not None:
            warnings.warn(
                "No brainmask can be applied with text input, ignoring it.",
            )
            args.mask = None

        if input_type == "4Dimage" and args.mask is not None:
            if args.mask == "BIDS":
                warnings.warn(
                    "BIDS masks cannot be applied with 4D image input, ignoring it.",
                )
                args.mask = None
            elif ("nii" in args.bids_dir and "gii" in args.mask) or (
                "gii" in args.bids_dir and "nii" in args.mask
            ):
                parser.error(
                    "The mask file should be of the same type as the input file (NIfTI or GIfTI)"
                )

        if args.mask is not None and args.mask != "BIDS":
            args.mask = op.abspath(args.mask)
            if not args.mask.endswith((".nii", ".nii.gz", ".gii", ".gii.gz")):
                parser.error(
                    "The mask file should be of the same type as the input file (NIfTI or GIfTI)"
                )
            if not op.isfile(args.mask):
                parser.error(
                    "The mask file provided does not exist, please provide a valid path."
                )

        if args.temporal_mask is not None:
            try:
                f = open(args.temporal_mask, "r")
                for line in f:
                    for each in line:
                        if each in ["0", "1"]:
                            temporal_mask.append(int(each))
            except:
                parser.error(
                    "Unable to read temporal mask file. Please make sure the file is a text file, consisting of a sequence of 0s and 1s of the same length as the signal"
                )

        if input_type != "BIDS":
            if para["TR"] <= 0:
                if input_type == "text":
                    parser.error("Please supply a valid TR using -TR argument")
                else:  # it's 4D image
                    if ".nii" in args.bids_dir:
                        TR = (spm_dep.spm.spm_vol(args.bids_dir).header.get_zooms())[-1]
                    else:
                        parser.error("Please supply a valid TR using -TR argument")
                    if TR <= 0:
                        parser.error("Please supply a valid TR using -TR argument")
                    else:
                        print(
                            "Invalid or no TR supplied, using implicit TR: {0}".format(
                                TR
                            ),
                            file=sys.stderr,
                        )
                        para["TR"] = TR
            para["dt"] = para["TR"] / para["T"]
            para["lag"] = np.arange(
                np.trunc(para["min_onset_search"] / para["dt"]),
                np.trunc(para["max_onset_search"] / para["dt"]) + 1,
                dtype="int",
            )

            if input_type == "text":
                file_type = op.splitext(args.bids_dir)[-1]
                fourD_rsHRF.demo_rsHRF(
                    args.bids_dir,
                    None,
                    args.output_dir,
                    para,
                    args.n_jobs,
                    file_type,
                    mode="time-series",
                    temporal_mask=temporal_mask,
                    wiener=args.wiener,
                )
                return 0

            else:  # it's 4D image
                # carry analysis with input_file and atlas
                file_type = op.splitext(args.bids_dir)
                if file_type[-1] == ".gz":
                    file_type = op.splitext(file_type[-2])[-1] + file_type[-1]
                else:
                    file_type = file_type[-1]

                fourD_rsHRF.demo_rsHRF(
                    args.bids_dir,
                    args.mask,
                    args.output_dir,
                    para,
                    args.n_jobs,
                    file_type,
                    mode="input",
                    temporal_mask=temporal_mask,
                    wiener=args.wiener,
                )
                return 0

        else:  # it's BIDS
            utils.bids.write_derivative_description(args.bids_dir, args.output_dir)
            bids_dir = Path(args.bids_dir)
            fname = bids_dir / "dataset_description.json"

            if fname.exists():
                desc = json.loads(Path(fname).read_text())
                if "DataType" in desc:
                    if desc["DataType"] != "derivative":
                        parser.error(
                            "Input data is not a derivative dataset"
                            ' (DataType in dataset_description.json is not equal to "derivative")'
                        )

                else:
                    parser.error(
                        "DataType is not defined in the dataset_description.json file. Please make sure DataType is defined. "
                        "Information on the dataset_description.json file can be found online "
                        "(https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html"
                        "#derived-dataset-and-pipeline-description)"
                    )
            else:
                parser.error(
                    "Could not find dataset_description.json file. Please make sure the BIDS data "
                    "structure is present and correct. Datasets can be validated online "
                    "using the BIDS Validator (http://incf.github.io/bids-validator/)."
                )

            if para["TR"] >= 0:
                warnings.warn(
                    "Explicit TR value is ignored when input is BIDS, as TR will be "
                    "read from the metadata of the input files."
                )

            layout = BIDSLayout(
                args.bids_dir, validate=False, config=["bids", "derivatives"]
            )

            if args.participant_label:
                input_subjects = args.participant_label
                subjects_to_analyze = layout.get_subjects(subject=input_subjects)
            else:
                subjects_to_analyze = layout.get_subjects()

            if len(subjects_to_analyze) == 0:
                parser.error(
                    "Could not find participants. Please make sure the BIDS data "
                    "structure is present and correct. Datasets can be validated online "
                    "using the BIDS Validator (http://incf.github.io/bids-validator/)."
                )

            if (
                args.mask is not None
                and args.mask != "BIDS"
                and not args.mask.endswith((".nii", ".nii.gz"))
            ):
                parser.error("Mask for BIDS input should end with .nii or .nii.gz")

            if args.bids_filter_file is not None:
                filter_list = json.loads(Path(args.bids_filter_file).read_text())

                default_input = {
                    "extension": "nii.gz",
                    "datatype": "func",
                    "desc": "preproc",
                    "task": "rest",
                    "suffix": "bold",
                }
                default_input["subject"] = subjects_to_analyze
                default_input.update(filter_list["bold"])

                all_inputs = layout.get(return_type="filename", **default_input)

                if args.mask == "BIDS":
                    default_mask = {
                        "extension": "nii.gz",
                        "datatype": "func",
                        "desc": "brain",
                        "task": "rest",
                        "suffix": "mask",
                    }
                    default_mask["subject"] = subjects_to_analyze
                    default_mask.update(filter_list["mask"])

                    all_masks = layout.get(return_type="filename", **default_mask)

            else:
                all_inputs = layout.get(
                    return_type="filename",
                    datatype="func",
                    subject=subjects_to_analyze,
                    task="rest",
                    desc="preproc",
                    suffix="bold",
                    extension=["nii", "nii.gz"],
                )
                if args.mask == "BIDS":
                    all_masks = layout.get(
                        return_type="filename",
                        datatype="func",
                        subject=subjects_to_analyze,
                        task="rest",
                        desc="brain",
                        suffix="mask",
                        extension=["nii", "nii.gz"],
                    )

            if not all_inputs != []:
                parser.error(
                    "There are no files of type *bold.nii / *bold.nii.gz "
                    "Please make sure to have at least one file of the above type "
                    "in the BIDS specification"
                )
            all_inputs.sort()

            if args.mask == "BIDS":
                if not all_masks != []:
                    parser.error(
                        "There are no files of type *mask.nii / *mask.nii.gz "
                        "Please make sure to have at least one file of the above type "
                        "in the BIDS specification"
                    )
                if len(all_inputs) != len(all_masks):
                    parser.error(
                        "The number of *bold.nii / .nii.gz and the number of "
                        "*mask.nii / .nii.gz are different. Please make sure that "
                        "there is one mask for each input_file present"
                    )

                all_masks.sort()

                all_prefix_match = False

                prefix_match_count = 0
                for i in range(len(all_inputs)):
                    input_prefix = all_inputs[i].split("/")[-1].split("_desc")[0]
                    mask_prefix = all_masks[i].split("/")[-1].split("_desc")[0]
                    if input_prefix == mask_prefix:
                        prefix_match_count += 1
                    else:
                        all_prefix_match = False
                        break
                if prefix_match_count == len(all_inputs):
                    all_prefix_match = True

                if not all_prefix_match:
                    parser.error(
                        "The mask and input files should have the same prefix for correspondence. "
                        "Please consider renaming your files"
                    )

            num_errors = 0
            for file_count in range(len(all_inputs)):
                file_type = all_inputs[file_count].split("bold")[1]
                if file_type == ".nii" or file_type == ".nii.gz":
                    try:
                        TR = layout.get_metadata(all_inputs[file_count])[
                            "RepetitionTime"
                        ]
                    except KeyError as e:
                        TR = spm_dep.spm.spm_vol(
                            all_inputs[file_count]
                        ).header.get_zooms()[-1]
                    para["TR"] = TR
                else:
                    spm_dep.spm.spm_vol(all_inputs[file_count])
                    TR = (
                        spm_dep.spm.spm_vol(all_inputs[file_count])
                        .get_arrays_from_intent("NIFTI_INTENT_TIME_SERIES")[0]
                        .meta.get_metadata()["TimeStep"]
                    )
                    para["TR"] = float(TR) * 0.001

                para["dt"] = para["TR"] / para["T"]
                para["lag"] = np.arange(
                    np.trunc(para["min_onset_search"] / para["dt"]),
                    np.trunc(para["max_onset_search"] / para["dt"]) + 1,
                    dtype="int",
                )
                num_errors += 1
                try:
                    fourD_rsHRF.demo_rsHRF(
                        all_inputs[file_count],
                        all_masks[file_count] if args.mask == "BIDS" else args.mask,
                        args.output_dir,
                        para,
                        args.n_jobs,
                        file_type,
                        mode="bids" + (" w/ atlas" if args.mask != "BIDS" else ""),
                        temporal_mask=temporal_mask,
                        wiener=args.wiener,
                    )
                    num_errors -= 1
                except ValueError as err:
                    print(err.args[0])
                except:
                    print("Unexpected error:", sys.exc_info()[0])
            success = len(all_inputs) - num_errors
            if success == 0:
                raise RuntimeError(
                    "Dimensions were inconsistent for all input-mask pairs; \n"
                    "No inputs were processed!"
                )
            return 0


def main():
    run_rsHRF()


if __name__ == "__main__":
    raise RuntimeError(
        "CLI.py should not be run directly;\n"
        "Please `pip install` rsHRF and use the `rsHRF` command"
    )
