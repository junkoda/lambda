dtfe_A
======

Taruya A term computed from DTFE grids

$ A_{DD}(k) = \int d^3 k e^-ikx < delta(x) delta(y) [u(x) - u(y)] > $

$ A_{DU}(k) = \int d^3 k e^-ikx < u'(x) delta(y) [u(x) - u(y)] > $

$ A_{UU}(k) = \int d^3 k e^-ikx < u'(x) u'(y) [u(x) - u(y)] > $

where,

$ u(x) = v_z(x)/(aH) $
$ u'(x) = -\partial u(x)/\partial z $

## File name

HDF5 file

```bash
<isnp>/taruya_bispectrum_<irealisation>.h5
```

2D grids of (k, mu) are saved in HDF5 files.

```text
Add[ik, imu]   A_DD(k, mu)
Adu[ik, imu]   A_DU(k, mu)
Auu[ik, imu]   A_UU(k, mu)
k[ik, imu]     mean k in 2D bin
mu[ik, imu]    meann mu in 2D bin
```

Array size is 100 x 10
bin width dk = 0.01 h/Mpc
          dmu = 0.1

## Example

```python
import h5py
with h5py.File('010/taruya_bispectrum_00001.h5', 'r') as f:
  ADD = f['Add'][:]
```
