set DIR (dirname (status --current-filename))
set ddm_time_view $DIR/ddm_time_view.py
set try2_data $DIR/../../data/initial-dev/try2-diff/auto_correlate_data_lin_diff_diff.npy
set -x FIGURE_PATH $PART_III_PROJECT_PATH/plots/try2_xaxis_fits
    #2>/dev/null
	#-np
python $ddm_time_view $try2_data -tm 130 -ij 0 2 \
    -p0 2.0  -1.70 0.175 50 \
    -lb 1.9  -2.10 0.165 5 \
    -ub 2.1  -1.60 0.225 125 \
    -tf 5 58 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 3 \
    -p0 2.0  -2.13 0.205 50 \
    -lb 1.9  -2.50 0.190 5 \
    -ub 2.1  -1.80 0.225 125 \
    -tf 5 90 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 4 \
    -p0 2.03 -2.16 0.220 50 \
    -lb 1.9  -2.50 0.210 5 \
    -ub 2.1  -1.80 0.225 125 \
    -tf 5 80 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 5 \
    -p0 2.0  -2.17 0.240 50 \
    -lb 1.9  -2.60 0.190 5 \
    -ub 2.1  -1.80 0.300 125 \
    -tf 5 58 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 6 \
    -p0 2.0  -2.15 0.275 50 \
    -lb 1.9  -2.50 0.190 5 \
    -ub 2.1  -1.80 0.300 125 \
    -tf 5 58 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 7 \
    -p0 2.0  -1.90 0.300 50 \
    -lb 1.9  -2.50 0.250 5 \
    -ub 2.1  -1.80 0.350 125 \
    -tf 5 45 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 8 \
    -p0 2.0  -2.00 0.315 50 \
    -lb 1.9  -2.15 0.250 5 \
    -ub 2.1  -1.85 0.350 125 \
    -tf 5 31 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 9 \
    -p0 2.0  -1.75 0.350 50 \
    -lb 1.9  -2.15 0.280 5 \
    -ub 2.1  -1.65 0.400 125 \
    -tf 5 28 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 10 \
    -p0 2.0  -1.90 0.370 50 \
    -lb 1.9  -2.15 0.280 5 \
    -ub 2.1  -1.50 0.400 125 \
    -tf 5 25 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 11 \
    -p0 2.0  -1.90 0.390 50 \
    -lb 1.9  -2.15 0.280 5 \
    -ub 2.2  -1.25 0.420 125 \
    -tf 5 23 -s
python $ddm_time_view $try2_data -tm 130 -ij 0 12 \
    -p0 2.0  -2.00 0.390 50 \
    -lb 1.9  -2.15 0.280 5 \
    -ub 2.2  -0.80 0.420 125 \
    -tf 5 22 -s
