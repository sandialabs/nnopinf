#!/usr/bin/env pvpython
"""Render per-model PNGs for torsion.e files using ParaView.

Example:
  pvpython render_model_pngs.py --disp-mag --output-dir renders
"""

import argparse
import os
from pathlib import Path

import paraview.servermanager as sm
from paraview.simple import (
    ExodusIIReader,
    GetAnimationScene,
    GetActiveViewOrCreate,
    GetTimeKeeper,
    Show,
    ColorBy,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    Render,
    SaveScreenshot,
    HideScalarBarIfNotNeeded,
    ResetCamera,
    Delete,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ROM_DIRECTORY = SCRIPT_DIRECTORY.parent / "roms"
FOM_EXODUS = ROM_DIRECTORY.parent / "fom" / "torsion.e"


def parse_args():
    p = argparse.ArgumentParser(description="Render PNGs for each model's torsion.e file.")
    p.add_argument("--array", default=None, help="Point or cell array to color by")
    p.add_argument("--disp-mag", action="store_true", help="Color by displacement magnitude (auto-detects array)")
    p.add_argument("--disp-array", default=None, help="Displacement vector array name (overrides auto-detect)")
    p.add_argument("--component", type=int, default=None, help="Component index for vector/tensor arrays")
    p.add_argument("--representation", default="Surface", choices=["Surface", "Surface With Edges", "Wireframe"], help="Geometry representation")
    p.add_argument("--camera", default="iso", choices=["iso", "xy", "xz", "yz", "reset"], help="Camera preset for 3D view")
    p.add_argument("--time", type=float, default=None, help="Time value to render (e.g., 0.0025)")
    p.add_argument("--times", default=None, help="Comma-separated list of times (e.g., 0.0025,0.0050)")
    p.add_argument("--transparent", action="store_true", default=True, help="Save screenshot with transparent background")
    p.add_argument("--opaque", action="store_true", help="Force opaque background (overrides --transparent)")
    p.add_argument("--color-bar", action="store_true", default=False, help="Show color bar (scalar bar)")
    p.add_argument("--no-color-bar", action="store_true", help="Hide color bar")
    p.add_argument("--size", default="1600x1000", help="Screenshot size, e.g. 1600x1000")
    p.add_argument("--output-dir", default="renders", help="Directory for saved PNGs")
    p.add_argument("--rom-dims", default="4,8,16,32", help="Comma-separated ROM dimensions")
    p.add_argument("--fom-times", default=None, help="Comma-separated FOM-only times (e.g., 0,0.00125,0.0025)")
    return p.parse_args()


def _find_array_name(reader, preferred):
    if preferred:
        return preferred
    candidates = ["disp_", "displacement", "disp", "u", "U", "Udisp", "Displacement"]
    point_vars = list(reader.PointVariables)
    cell_vars = list(reader.ElementVariables)
    for name in candidates:
        if name in point_vars or name in cell_vars:
            return name
    return None


def _set_camera(view, reader, preset):
    if preset == "reset":
        ResetCamera(view)
        return
    info = reader.GetDataInformation()
    bounds = info.GetBounds()
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diag = (dx * dx + dy * dy + dz * dz) ** 0.5
    if diag == 0:
        ResetCamera(view)
        return
    if preset == "xy":
        view.CameraPosition = [cx, cy, cz + 1.5 * diag]
        view.CameraViewUp = [0, 1, 0]
    elif preset == "xz":
        view.CameraPosition = [cx, cy - 1.5 * diag, cz]
        view.CameraViewUp = [0, 0, 1]
    elif preset == "yz":
        view.CameraPosition = [cx + 1.5 * diag, cy, cz]
        view.CameraViewUp = [0, 0, 1]
    else:  # iso
        # Pull back more in Z to avoid clipping at the top of the model.
        view.CameraPosition = [cx + 1.2 * diag, cy - 1.2 * diag, cz + 1.05 * diag]
        view.CameraViewUp = [0, 0, 1]
    view.CameraFocalPoint = [cx, cy, cz]
    try:
        view.ResetCameraClippingRange()
    except AttributeError:
        pass


def configure_display(args, reader, view, array_name, color_range):
    display = Show(reader, view)
    display.Representation = args.representation

    if args.disp_mag:
        if array_name is None:
            raise SystemExit("Could not find a displacement array; use --disp-array to specify one.")
        try:
            ColorBy(display, ("POINT_DATA", array_name, "Magnitude"))
        except TypeError:
            ColorBy(display, ("POINT_DATA", array_name))
        if display.LookupTable is None:
            try:
                ColorBy(display, ("CELL_DATA", array_name, "Magnitude"))
            except TypeError:
                ColorBy(display, ("CELL_DATA", array_name))
        if display.LookupTable is not None and hasattr(display.LookupTable, "VectorMode"):
            display.LookupTable.VectorMode = "Magnitude"
        lut = GetColorTransferFunction(array_name)
        pwf = GetOpacityTransferFunction(array_name)
        display.LookupTable = lut
        display.OpacityTransferFunction = pwf
        if color_range is None:
            try:
                lut.RescaleTransferFunctionToDataRange()
            except TypeError:
                lut.RescaleTransferFunctionToDataRange(True, False)
        else:
            lut.RescaleTransferFunction(color_range[0], color_range[1])
        try:
            pwf.RescaleTransferFunctionToDataRange()
        except TypeError:
            pwf.RescaleTransferFunctionToDataRange(True, False)
        if not args.no_color_bar:
            display.SetScalarBarVisibility(view, True)
    elif args.array:
        ColorBy(display, ("POINT_DATA", args.array))
        if display.LookupTable is None:
            ColorBy(display, ("CELL_DATA", args.array))
        if args.component is not None:
            display.LookupTable.VectorMode = "Component"
            display.LookupTable.VectorComponent = args.component
        lut = GetColorTransferFunction(args.array)
        pwf = GetOpacityTransferFunction(args.array)
        display.LookupTable = lut
        display.OpacityTransferFunction = pwf
        if color_range is None:
            try:
                lut.RescaleTransferFunctionToDataRange()
            except TypeError:
                lut.RescaleTransferFunctionToDataRange(True, False)
        else:
            lut.RescaleTransferFunction(color_range[0], color_range[1])
        try:
            pwf.RescaleTransferFunctionToDataRange()
        except TypeError:
            pwf.RescaleTransferFunctionToDataRange(True, False)
        if not args.no_color_bar:
            display.SetScalarBarVisibility(view, True)
    else:
        if display.LookupTable is not None:
            HideScalarBarIfNotNeeded(display.LookupTable, view)
    return display


def _get_array_range(reader, array_name, use_magnitude):
    info = reader.GetDataInformation()
    point_info = info.GetPointDataInformation()
    cell_info = info.GetCellDataInformation()
    array = None
    if hasattr(point_info, "GetArray"):
        array = point_info.GetArray(array_name)
    if array is None and hasattr(cell_info, "GetArray"):
        array = cell_info.GetArray(array_name)
    if array is not None:
        if use_magnitude:
            try:
                return array.GetRange(-1)
            except TypeError:
                pass
        return array.GetRange(0)

    array_info = None
    if hasattr(point_info, "GetArrayInformation"):
        array_info = point_info.GetArrayInformation(array_name)
    if array_info is None and hasattr(cell_info, "GetArrayInformation"):
        array_info = cell_info.GetArrayInformation(array_name)
    if array_info is None:
        return None
    if use_magnitude and hasattr(array_info, "GetComponentRange"):
        try:
            return array_info.GetComponentRange(-1)
        except TypeError:
            pass
    if hasattr(array_info, "GetRange"):
        return array_info.GetRange(0)
    if hasattr(array_info, "GetComponentRange"):
        return array_info.GetComponentRange(0)
    return None


def _get_first_block(dataset):
    if hasattr(dataset, "GetPointData"):
        return dataset
    if hasattr(dataset, "NewIterator"):
        it = dataset.NewIterator()
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            block = it.GetCurrentDataObject()
            it.GoToNextItem()
            if block is None:
                continue
            if hasattr(block, "GetPointData"):
                return block
    return None


def _get_dataset_array_range(dataset, array_name, use_magnitude):
    data_obj = _get_first_block(dataset)
    if data_obj is None:
        return None
    point_data = data_obj.GetPointData()
    cell_data = data_obj.GetCellData()
    array = point_data.GetArray(array_name)
    if array is None:
        array = cell_data.GetArray(array_name)
    if array is None:
        return None
    if use_magnitude:
        try:
            return array.GetRange(-1)
        except TypeError:
            pass
    return array.GetRange(0)


def _compute_fom_color_range(args, view, fom_exodus, array_name, times):
    if not os.path.exists(fom_exodus):
        raise SystemExit(f"FOM exodus file not found: {fom_exodus}")
    global_min = None
    global_max = None
    reader = ExodusIIReader(FileName=[str(fom_exodus)])
    reader.PointVariables = list(reader.PointVariables)
    reader.ElementVariables = list(reader.ElementVariables)
    for time_value in times:
        reader.UpdatePipeline()
        if time_value is not None:
            animation_scene = GetAnimationScene()
            animation_scene.UpdateAnimationUsingDataTimeSteps()
            time_keeper = GetTimeKeeper()
            time_keeper.Time = time_value
            reader.UpdatePipeline(time_value)
        # Use actual data range via LUT to avoid metadata issues.
        display = configure_display(args, reader, view, array_name, None)
        lut = display.LookupTable
        if lut is not None:
            rng = None
            try:
                rng = lut.GetRange()
            except TypeError:
                try:
                    rng = lut.GetRange(0)
                except TypeError:
                    rng = None
            except AttributeError:
                    rng = None
            if rng is None:
                try:
                    lut.RescaleTransferFunctionToDataRange()
                except TypeError:
                    lut.RescaleTransferFunctionToDataRange(True, False)
                try:
                    rng = lut.GetRange()
                except TypeError:
                    try:
                        rng = lut.GetRange(0)
                    except TypeError:
                        rng = None
            if rng is not None:
                if global_min is None or rng[0] < global_min:
                    global_min = rng[0]
                if global_max is None or rng[1] > global_max:
                    global_max = rng[1]
        if lut is None or rng is None:
            dataset = sm.Fetch(reader)
            arr_range = _get_dataset_array_range(dataset, array_name, args.disp_mag)
            if arr_range is not None:
                if global_min is None or arr_range[0] < global_min:
                    global_min = arr_range[0]
                if global_max is None or arr_range[1] > global_max:
                    global_max = arr_range[1]
        Delete(display)
    Delete(reader)
    if global_min is None or global_max is None:
        raise SystemExit(
            "Could not determine FOM color range for the requested array."
        )
    return (global_min, global_max)


def render_exodus(args, view, exodus_path, output_path, array_name, color_range, time_value):
    reader = ExodusIIReader(FileName=[exodus_path])
    reader.PointVariables = list(reader.PointVariables)
    reader.ElementVariables = list(reader.ElementVariables)
    reader.UpdatePipeline()

    if time_value is not None:
        animation_scene = GetAnimationScene()
        animation_scene.UpdateAnimationUsingDataTimeSteps()
        time_keeper = GetTimeKeeper()
        time_keeper.Time = time_value
        reader.UpdatePipeline(time_value)

    display = configure_display(args, reader, view, array_name, color_range)
    _set_camera(view, reader, args.camera)
    Render()

    w, h = (int(x) for x in args.size.lower().split("x"))
    transparent = args.transparent and not args.opaque
    SaveScreenshot(
        output_path,
        view,
        ImageResolution=[w, h],
        TransparentBackground=1 if transparent else 0,
    )

    Delete(display)
    Delete(reader)


def main():
    args = parse_args()
    rom_dims = [int(x) for x in args.rom_dims.split(",") if x.strip()]
    model_runs = [
        ("NN-OpInf-SPSD-Potential", "spsd-potential", "torsion.e"),
        ("OpInf-A", "linear", "torsion-linear.e"),
        ("OpInf-AH", "quadratic", "torsion-quadratic.e"),
        ("NN-OpInf-NN", "vanilla", "torsion-vanilla.e"),
        ("NN-OpInf-SpML-Lagrangian", "lpopinf", "torsion-lpopinf.e"),
        ("NN-OpInf-Linear-Lagrangian", "linear-lagrangian", "torsion-linear-lagrangian.e"),
    ]
    fom_exodus = FOM_EXODUS

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    view = GetActiveViewOrCreate("RenderView")

    if args.times:
        times = [float(x) for x in args.times.split(",") if x.strip()]
    elif args.time is not None:
        times = [args.time]
    else:
        times = [None]

    array_name = args.disp_array or args.array
    if args.disp_mag and array_name is None:
        for label, directory_suffix, exodus_filename in model_runs:
            for rom_dim in rom_dims:
                exodus_path = ROM_DIRECTORY / directory_suffix / f"dim{rom_dim}" / exodus_filename
                if not os.path.exists(exodus_path):
                    continue
                reader = ExodusIIReader(FileName=[str(exodus_path)])
                reader.PointVariables = list(reader.PointVariables)
                reader.ElementVariables = list(reader.ElementVariables)
                reader.UpdatePipeline()
                array_name = _find_array_name(reader, None)
                Delete(reader)
                if array_name is not None:
                    break
            if array_name is not None:
                break
        if array_name is None:
            raise SystemExit("Could not find a displacement array; use --disp-array to specify one.")

    if args.disp_mag or args.array:
        color_range = _compute_fom_color_range(args, view, fom_exodus, array_name, times)
    else:
        color_range = None

    # Render ROM models.
    for label, directory_suffix, exodus_filename in model_runs:
        for rom_dim in rom_dims:
            exodus_path = ROM_DIRECTORY / directory_suffix / f"dim{rom_dim}" / exodus_filename
            if not os.path.exists(exodus_path):
                print(f"Skipping missing file: {exodus_path}")
                continue
            for time_value in times:
                time_tag = "tNone"
                if time_value is not None:
                    time_tag = f"t{time_value:.4f}"
                output_name = f"{label}-{rom_dim}-{time_tag}.png"
                output_path = os.path.join(output_dir, output_name)
                print(f"Rendering {exodus_path} @ {time_value} -> {output_path}")
                render_exodus(args, view, str(exodus_path), output_path, array_name, color_range, time_value)

    # Render FOM for the main times list.
    if os.path.exists(fom_exodus):
        for time_value in times:
            time_tag = "tNone"
            if time_value is not None:
                time_tag = f"t{time_value:.4f}"
            output_name = f"fom-{time_tag}.png"
            output_path = os.path.join(output_dir, output_name)
            print(f"Rendering {fom_exodus} @ {time_value} -> {output_path}")
            render_exodus(args, view, str(fom_exodus), output_path, array_name, color_range, time_value)
    else:
        print(f"Skipping missing file: {fom_exodus}")

    if args.fom_times:
        fom_times = [float(x) for x in args.fom_times.split(",") if x.strip()]
        for time_value in fom_times:
            time_tag = f"t{time_value:.4f}"
            output_name = f"fom-{time_tag}.png"
            output_path = os.path.join(output_dir, output_name)
            print(f"Rendering {fom_exodus} @ {time_value} -> {output_path}")
            render_exodus(args, view, str(fom_exodus), output_path, array_name, color_range, time_value)


if __name__ == "__main__":
    main()
