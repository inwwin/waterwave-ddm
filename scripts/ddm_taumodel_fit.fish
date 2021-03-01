set DIR (dirname (status --current-filename))
set ddm_time_view $DIR/ddm_time_view.py
set try2_data $DIR/../../data/initial-dev/try2-diff/auto_correlate_data_lin_diff_diff.npy
    #2>/dev/null
	#-np
python $ddm_time_view $try2_data -tm 130 -ij 0 2 \
    -p0 2.0  -1.70 0.175 70 \
    -lb 1.9  -2.10 0.165 55 \
    -ub 2.1  -1.60 0.225 110 \
    -tf 5 70
python $ddm_time_view $try2_data -tm 130 -ij 0 3 \
    -p0 2.0  -2.13 0.205 81 \
    -lb 1.9  -2.50 0.190 70 \
    -ub 2.1  -1.80 0.225 110 \
    -tf 5 120
python $ddm_time_view $try2_data -tm 130 -ij 0 4 \
	-p0 2.03 -2.16 0.220 93 \
	-lb 1.9  -2.50 0.210 80 \
	-ub 2.1  -1.80 0.225 110 \
	-tf 5 120
python $ddm_time_view $try2_data -tm 130 -ij 0 5 \
    -p0 2.0  -2.17 0.240 49 \
    -lb 1.9  -2.50 0.190 40 \
    -ub 2.1  -1.80 0.300 110 \
    -tf 5 80
python $ddm_time_view $try2_data -tm 130 -ij 0 6 \
    -p0 2.0  -2.15 0.275 54 \
    -lb 1.9  -2.50 0.190 40 \
    -ub 2.1  -1.80 0.300 110 \
    -tf 5 70
python $ddm_time_view $try2_data -tm 130 -ij 0 7 \
    -p0 2.0  -1.90 0.300 45 \
    -lb 1.9  -2.50 0.250 40 \
    -ub 2.1  -1.80 0.350 110 \
    -tf 5 55
python $ddm_time_view $try2_data -tm 130 -ij 0 8 \
    -p0 2.0  -2.00 0.315 20 \
    -lb 1.9  -2.15 0.250 10 \
    -ub 2.1  -1.85 0.350 30 \
    -tf 5 40
python $ddm_time_view $try2_data -tm 130 -ij 0 9 \
    -p0 2.0  -1.75 0.350 22 \
    -lb 1.9  -2.15 0.280 10 \
    -ub 2.1  -1.65 0.400 40 \
    -tf 5 35
python $ddm_time_view $try2_data -tm 130 -ij 0 10 \
    -p0 2.0  -1.90 0.370 14 \
    -lb 1.9  -2.15 0.280 10 \
    -ub 2.1  -1.65 0.400 40 \
    -tf 5 28
python $ddm_time_view $try2_data -tm 130 -ij 0 11 \
    -p0 2.0  -1.90 0.390 10 \
    -lb 1.9  -2.15 0.280 5 \
    -ub 2.1  -1.65 0.420 20 \
    -tf 5 26
python $ddm_time_view $try2_data -tm 130 -ij 0 12 \
    -p0 2.0  -2.00 0.390 5 \
    -lb 1.9  -2.15 0.280 2 \
    -ub 2.1  -1.65 0.420 10 \
    -tf 5 20
