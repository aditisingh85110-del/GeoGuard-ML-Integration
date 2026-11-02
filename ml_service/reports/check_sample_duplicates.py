import h5py
import numpy as np

for i in [1, 2, 5, 10]:
    t_path = f"ml_service/data/raw/train/images/image_{i}.h5"
    v_path = f"ml_service/data/raw/validation/images/image_{i}.h5"
    te_path = f"ml_service/data/raw/test/images/image_{i}.h5"
    
    with h5py.File(t_path, "r") as ft:
        dt = ft["img"][:]
    with h5py.File(v_path, "r") as fv:
        dv = fv["img"][:]
    with h5py.File(te_path, "r") as fte:
        dte = fte["img"][:]
        
    print(f"Sample {i}:")
    print(f"  Train mean: {dt.mean():.4f}, Val mean: {dv.mean():.4f}, Test mean: {dte.mean():.4f}")
    print(f"  Train == Val array equal: {np.array_equal(dt, dv)}")
    print(f"  Train == Test array equal: {np.array_equal(dt, dte)}")
    print(f"  Val == Test array equal: {np.array_equal(dv, dte)}")
