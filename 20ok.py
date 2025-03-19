import os

#Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L2
from sklearn.metrics import explained_variance_score, r2_score, max_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10
import seaborn as sns
import io
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, sosfilt
from sklearn.preprocessing import MinMaxScaler
import pywt
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt
from scipy.signal import cwt, morlet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import StringIO

# Set styles for plotting
sns.set_style("whitegrid")
plt.style.use("ggplot")

# Bitcoin and Ethereum data
bitcoin_data = """unix,date,symbol,open,high,low,close,Volume USD,Volume BTC
1725062400000,2024-08-31 00:00:00,BTC/USD,59300,59622,58932,59155,48067339.5838566,812.56596372
1724976000000,2024-08-30 00:00:00,BTC/USD,59512,60102,57876,59300,112697030.870808,1900.45583256
1724889600000,2024-08-29 00:00:00,BTC/USD,59156,61290,58885,59512,59777620.1863308,1004.46330465
1724803200000,2024-08-28 00:00:00,BTC/USD,59540,60330,58085,59156,46172751.28910544,780.52524324
1724716800000,2024-08-27 00:00:00,BTC/USD,62966,63307,58200,59545,68861244.17943025,1156.45720345
1724630400000,2024-08-26 00:00:00,BTC/USD,64362,64579,62934,62966,24803193.1962431,393.91406785
1724544000000,2024-08-25 00:00:00,BTC/USD,64236,65200,63912,64361,15615238.636488039,242.61957764
1724457600000,2024-08-24 00:00:00,BTC/USD,64140,64609,63667,64230,16788053.6636457,261.37402559
1724371200000,2024-08-23 00:00:00,BTC/USD,60483,65074,60461,64151,48095375.81600226,749.72137326
1724284800000,2024-08-22 00:00:00,BTC/USD,61253,61467,59842,60483,21273847.75218096,351.73268112
1724198400000,2024-08-21 00:00:00,BTC/USD,59172,61898,58957,61252,60141538.71213604,981.87061177
1724112000000,2024-08-20 00:00:00,BTC/USD,59578,61472,58718,59172,65337396.6788502,1104.19449535
1724025600000,2024-08-19 00:00:00,BTC/USD,58533,59700,57936,59588,17648828.559064522,296.18091829
1723939200000,2024-08-18 00:00:00,BTC/USD,59593,60313,58533,58543,29521690.885747142,504.27362598
1723852800000,2024-08-17 00:00:00,BTC/USD,59006,59800,58950,59589,7851987.0288054,131.7690686
1723766400000,2024-08-16 00:00:00,BTC/USD,57594,59905,57171,58996,26431190.05272564,448.01664609
1723680000000,2024-08-15 00:00:00,BTC/USD,58765,59909,56220,57594,46706940.408227935,810.96885801
1723593600000,2024-08-14 00:00:00,BTC/USD,60666,61898,58558,58777,49272170.6599422,838.2899886
1723507200000,2024-08-13 00:00:00,BTC/USD,59440,61635,58527,60666,46999656.88260504,774.72813244
1723420800000,2024-08-12 00:00:00,BTC/USD,58816,60764,57720,59440,44366771.761768006,746.4127147
1723334400000,2024-08-11 00:00:00,BTC/USD,61019,61916,58430,58817,22084180.961430937,375.47275382
1723248000000,2024-08-10 00:00:00,BTC/USD,60926,61535,60342,61018,10537404.53941418,172.69337801
1723161600000,2024-08-09 00:00:00,BTC/USD,61771,61809,59630,60942,22721095.2527727,372.83146685
1723075200000,2024-08-08 00:00:00,BTC/USD,55195,62770,54847,61801,93581723.61319607,1514.24287007
1722988800000,2024-08-07 00:00:00,BTC/USD,56142,57833,54659,55179,46500913.02281157,842.72844783
1722902400000,2024-08-06 00:00:00,BTC/USD,54179,57207,54142,56187,65688964.23536541,1169.11321543
1722816000000,2024-08-05 00:00:00,BTC/USD,58203,58344,49130,54158,392739593.09807014,7251.73738133
1722729600000,2024-08-04 00:00:00,BTC/USD,60821,61213,57300,58220,39248312.494049,674.13796795
1722643200000,2024-08-03 00:00:00,BTC/USD,61548,62260,60000,60825,53338645.602077246,876.91977973
1722556800000,2024-08-02 00:00:00,BTC/USD,65462,65668,61291,61529,69975177.76355535,1137.27149415
1722470400000,2024-08-01 00:00:00,BTC/USD,64724,65720,62409,65465,93800731.29178931,1432.83787202
1722384000000,2024-07-31 00:00:00,BTC/USD,66348,66947,64630,64724,31940066.05691924,493.48102801
1722297600000,2024-07-30 00:00:00,BTC/USD,66927,67179,65556,66349,45300557.565298416,682.76172309
1722211200000,2024-07-29 00:00:00,BTC/USD,68417,70162,66666,66933,81408720.95297587,1216.27180842
1722124800000,2024-07-28 00:00:00,BTC/USD,68020,68443,67223,68419,10875430.94291511,158.95337469
1722038400000,2024-07-27 00:00:00,BTC/USD,68061,69529,66858,68023,51899877.18716509,762.97542283
1721952000000,2024-07-26 00:00:00,BTC/USD,65867,68333,65850,68061,51009120.42396555,749.46181255
1721865600000,2024-07-25 00:00:00,BTC/USD,65503,66241,63532,65849,71995478.31346898,1093.34201451
1721779200000,2024-07-24 00:00:00,BTC/USD,66101,67241,65229,65482,48169804.0390128,735.6190104
1721692800000,2024-07-23 00:00:00,BTC/USD,67691,67906,65547,66101,37346098.68468423,564.98538123
1721606400000,2024-07-22 00:00:00,BTC/USD,68269,68560,66715,67698,48088067.04968976,710.33216712
1721520000000,2024-07-21 00:00:00,BTC/USD,67239,68479,65944,68265,49725175.174308896,728.41390426
1721433600000,2024-07-20 00:00:00,BTC/USD,66786,67694,66368,67239,28711911.58622163,427.01276917
1721347200000,2024-07-19 00:00:00,BTC/USD,64047,67503,63414,66789,67651134.78564711,1012.90833499
1721260800000,2024-07-18 00:00:00,BTC/USD,64193,65200,63292,64048,39439375.20402048,615.77840376
1721174400000,2024-07-17 00:00:00,BTC/USD,65180,66180,63958,64204,32753408.05764524,510.14591081
1721088000000,2024-07-16 00:00:00,BTC/USD,64883,65488,62537,65188,54745712.66435448,839.81273646
1721001600000,2024-07-15 00:00:00,BTC/USD,60933,65014,60840,64882,87171896.92124258,1343.54515769
1720915200000,2024-07-14 00:00:00,BTC/USD,59381,61535,59379,60946,63017957.67828728,1033.99661468
1720828800000,2024-07-13 00:00:00,BTC/USD,58055,60063,57932,59382,37035028.88769546,623.67432703
1720742400000,2024-07-12 00:00:00,BTC/USD,57457,58653,56668,58054,65253542.35533304,1124.01457876
1720656000000,2024-07-11 00:00:00,BTC/USD,57909,59654,57237,57458,78756856.56092179,1370.68565841
1720569600000,2024-07-10 00:00:00,BTC/USD,58205,59606,57353,57908,55977181.4807098,966.65713685
1720483200000,2024-07-09 00:00:00,BTC/USD,56873,58422,56462,58205,76109939.09210336,1307.61857387
1720396800000,2024-07-08 00:00:00,BTC/USD,56035,58350,54500,56873,96632115.16610241,1699.0859488
1720310400000,2024-07-07 00:00:00,BTC/USD,58326,58563,55929,56039,54088258.40401318,965.18957162
1720224000000,2024-07-06 00:00:00,BTC/USD,56759,58538,56188,58326,87042453.94919263,1492.34396237
1720137600000,2024-07-05 00:00:00,BTC/USD,57118,57615,53219,56759,171344955.5877077,3018.81561669
1720051200000,2024-07-04 00:00:00,BTC/USD,60298,60550,56810,57118,137008655.2216619,2398.69489866
1719964800000,2024-07-03 00:00:00,BTC/USD,62217,62386,59476,60298,60697523.634062,1006.625819
1719878400000,2024-07-02 00:00:00,BTC/USD,62939,63331,61878,62219,38071734.65205573,611.89885167
1719792000000,2024-07-01 00:00:00,BTC/USD,62867,63911,62628,62953,43677397.8640208,693.8096336
1719705600000,2024-06-30 00:00:00,BTC/USD,61084,63131,60825,62867,39158463.773117155,622.87788145
1719619200000,2024-06-29 00:00:00,BTC/USD,60546,61352,60513,61087,48742436.03082451,797.91831373
1719532800000,2024-06-28 00:00:00,BTC/USD,61804,62295,60151,60547,53701811.53999985,886.94421755
1719446400000,2024-06-27 00:00:00,BTC/USD,60930,62499,60708,61825,81442045.05897374,1317.29955615
1719360000000,2024-06-26 00:00:00,BTC/USD,61878,62558,60789,60923,53065425.78042902,871.02450274
1719273600000,2024-06-25 00:00:00,BTC/USD,60371,62470,60340,61878,96528050.94222546,1559.97367307
1719187200000,2024-06-24 00:00:00,BTC/USD,63292,63461,58555,60372,178122474.393318,2950.4153315
1719100800000,2024-06-23 00:00:00,BTC/USD,64369,64618,63290,63292,36004977.03191432,568.87090046
1719014400000,2024-06-22 00:00:00,BTC/USD,64240,64613,64062,64370,32713132.786229003,508.2046417
1718928000000,2024-06-21 00:00:00,BTC/USD,65008,65196,63500,64237,72083371.24992858,1122.14722434
1718841600000,2024-06-20 00:00:00,BTC/USD,65071,66585,64654,65007,75242413.82279097,1157.45094871
1718755200000,2024-06-19 00:00:00,BTC/USD,65257,65820,64784,65071,46322969.52038625,711.88347375
1718668800000,2024-06-18 00:00:00,BTC/USD,66569,66623,64179,65253,104719427.93984985,1604.82166245
1718582400000,2024-06-17 00:00:00,BTC/USD,66677,67302,65147,66569,45399753.27441128,681.99542241
1718496000000,2024-06-16 00:00:00,BTC/USD,66229,66947,66054,66677,22445093.17228191,336.62422083
1718409600000,2024-06-15 00:00:00,BTC/USD,66039,66442,65886,66229,9660666.008919,145.867611
1718323200000,2024-06-14 00:00:00,BTC/USD,66776,67370,65088,66038,53768348.301872544,814.2031603300001
1718236800000,2024-06-13 00:00:00,BTC/USD,68266,68444,66271,66776,35107776.38667032,525.75440857
1718150400000,2024-06-12 00:00:00,BTC/USD,67330,70037,66919,68266,79286510.55601826,1161.43483661
1718064000000,2024-06-11 00:00:00,BTC/USD,69530,69552,66150,67331,90092080.91681732,1338.04756972
1717977600000,2024-06-10 00:00:00,BTC/USD,69645,70150,69170,69530,33770940.8427352,485.70316184
1717891200000,2024-06-09 00:00:00,BTC/USD,69337,69798,69101,69646,33972284.3235859,487.78514665
1717804800000,2024-06-08 00:00:00,BTC/USD,69393,69618,69204,69338,8920820.19759616,128.65701632
1717718400000,2024-06-07 00:00:00,BTC/USD,70835,71950,68464,69394,119855151.04656912,1727.16879048
1717632000000,2024-06-06 00:00:00,BTC/USD,71106,71651,70155,70836,47298802.9210734,667.72266815
1717545600000,2024-06-05 00:00:00,BTC/USD,70594,71717,70409,71101,61305707.41102738,862.23410938
1717459200000,2024-06-04 00:00:00,BTC/USD,68852,71071,68607,70604,56701964.33775384,803.09846946
1717372800000,2024-06-03 00:00:00,BTC/USD,67812,70294,67625,68859,82901465.54763697,1203.93072144
1717286400000,2024-06-02 00:00:00,BTC/USD,67771,68444,67340,67816,11912802.392428001,175.6635955
1717200000000,2024-06-01 00:00:00,BTC/USD,67573,67890,67463,67759,16386598.74827047,241.83649033"""

ethereum_data = """unix,date,symbol,open,high,low,close,Volume USD,Volume ETH
1725062400000,2024-08-31 00:00:00,ETH/USD,2533.3,2540.0,2500.1,2518.2,3490019.1440292415,1385.91817331
1724976000000,2024-08-30 00:00:00,ETH/USD,2533.4,2556.1,2441.0,2532.4,11477354.079419801,4532.2042645
1724889600000,2024-08-29 00:00:00,ETH/USD,2532.9,2601.1,2512.7,2535.6,3013033.916426064,1188.29228444
1724803200000,2024-08-28 00:00:00,ETH/USD,2460.2,2560.6,2424.0,2533.1,24901286.2302736,9830.36051884
1724716800000,2024-08-27 00:00:00,ETH/USD,2684.9,2704.3,2398.9,2463.3,29511505.97604021,11980.47577479
1724630400000,2024-08-26 00:00:00,ETH/USD,2752.9,2766.4,2672.5,2685.5,5090989.484082115,1895.73244613
1724544000000,2024-08-25 00:00:00,ETH/USD,2771.8,2796.4,2739.3,2748.8,1848196.644871616,672.36490282
1724457600000,2024-08-24 00:00:00,ETH/USD,2767.7,2823.3,2737.8,2771.1,5967837.699024943,2153.59882322
1724371200000,2024-08-23 00:00:00,ETH/USD,2627.3,2803.6,2627.3,2766.3,12076653.628368342,4365.63410634
1724284800000,2024-08-22 00:00:00,ETH/USD,2635.6,2648.2,2589.8,2628.2,24498531.57459069,9321.41068967
1724198400000,2024-08-21 00:00:00,ETH/USD,2580.0,2666.7,2540.3,2634.6,7304201.084120675,2772.41368106
1724112000000,2024-08-20 00:00:00,ETH/USD,2641.7,2699.4,2560.9,2580.5,26036564.65793591,10089.73635262
1724025600000,2024-08-19 00:00:00,ETH/USD,2617.0,2653.0,2569.4,2645.2,5281542.947701336,1996.65165118
1723939200000,2024-08-18 00:00:00,ETH/USD,2619.2,2692.1,2600.6,2614.9,4516920.784628745,1727.37802005
1723852800000,2024-08-17 00:00:00,ETH/USD,2598.8,2632.9,2594.1,2619.0,2320614.35049465,886.06886235
1723766400000,2024-08-16 00:00:00,ETH/USD,2572.8,2634.3,2555.3,2598.3,4610159.277333094,1774.29830171
1723680000000,2024-08-15 00:00:00,ETH/USD,2665.0,2678.1,2521.7,2570.9,10187193.463750573,3962.50086108
1723593600000,2024-08-14 00:00:00,ETH/USD,2707.1,2784.6,2637.1,2667.4,4350614.929034018,1631.03206457
1723507200000,2024-08-13 00:00:00,ETH/USD,2727.9,2742.7,2615.4,2703.1,7301434.366503775,2701.13364896
1723420800000,2024-08-12 00:00:00,ETH/USD,2558.9,2757.0,2514.5,2727.3,13324599.407388901,4885.637593
1723334400000,2024-08-11 00:00:00,ETH/USD,2614.0,2723.5,2546.5,2557.8,5587560.850240656,2184.51827752
1723248000000,2024-08-10 00:00:00,ETH/USD,2603.0,2647.3,2581.3,2612.1,2725225.95052542,1043.3084302
1723161600000,2024-08-09 00:00:00,ETH/USD,2688.3,2710.3,2556.6,2604.4,6385873.603325137,2451.95576844
1723075200000,2024-08-08 00:00:00,ETH/USD,2345.5,2725.7,2325.6,2687.5,15843414.896863125,5895.22414767
1722988800000,2024-08-07 00:00:00,ETH/USD,2467.6,2556.9,2313.6,2345.9,8431055.064205004,3593.95330756
1722902400000,2024-08-06 00:00:00,ETH/USD,2425.9,2559.9,2423.3,2466.4,10113405.571164018,4100.47257994
1722816000000,2024-08-05 00:00:00,ETH/USD,2692.5,2700.1,2099.9,2423.7,60569433.78866832,24990.483058410002
1722729600000,2024-08-04 00:00:00,ETH/USD,2908.5,2940.3,2629.1,2691.0,18380626.06270644,6830.40730684
1722643200000,2024-08-03 00:00:00,ETH/USD,2992.9,3021.2,2865.1,2910.0,13259041.446879301,4556.37163123
1722556800000,2024-08-02 00:00:00,ETH/USD,3209.0,3223.0,2973.0,2993.8,14230017.867441894,4753.16249163
1722470400000,2024-08-01 00:00:00,ETH/USD,3237.0,3246.8,3086.3,3210.1,9808174.259350166,3055.41081566
1722384000000,2024-07-31 00:00:00,ETH/USD,3286.3,3354.8,3219.7,3239.4,6214456.362178884,1918.39734586
1722297600000,2024-07-30 00:00:00,ETH/USD,3324.3,3372.5,3240.3,3285.7,3279014.401159376,997.96524368
1722211200000,2024-07-29 00:00:00,ETH/USD,3277.7,3403.7,3264.2,3327.9,11130730.089634044,3344.67084036
1722124800000,2024-07-28 00:00:00,ETH/USD,3253.9,3292.5,3205.8,3280.2,3225764.5160839558,983.40482778
1722038400000,2024-07-27 00:00:00,ETH/USD,3282.5,3333.0,3202.0,3254.3,6665398.1065889,2048.181823
1721952000000,2024-07-26 00:00:00,ETH/USD,3179.0,3292.7,3177.1,3281.7,8837470.416509978,2692.95499787
1721865600000,2024-07-25 00:00:00,ETH/USD,3342.6,3347.7,3093.9,3178.9,22142449.422530953,6965.44383986
1721779200000,2024-07-24 00:00:00,ETH/USD,3490.1,3501.9,3304.2,3342.5,24828527.363684975,7428.13084927
1721692800000,2024-07-23 00:00:00,ETH/USD,3446.5,3548.1,3395.6,3488.1,10507532.978425097,3012.39442058
1721606400000,2024-07-22 00:00:00,ETH/USD,3542.5,3566.4,3433.3,3448.1,5508405.412582072,1597.51904312
1721520000000,2024-07-21 00:00:00,ETH/USD,3522.6,3551.7,3417.6,3541.5,9311560.18763562,2629.27013628
1721433600000,2024-07-20 00:00:00,ETH/USD,3510.3,3544.5,3487.7,3524.6,4939408.487495016,1401.40965996
1721347200000,2024-07-19 00:00:00,ETH/USD,3430.2,3546.6,3382.1,3510.5,7549770.812800195,2150.62549859
1721260800000,2024-07-18 00:00:00,ETH/USD,3391.2,3493.2,3372.4,3436.5,3436374.768338505,999.96355837
1721174400000,2024-07-17 00:00:00,ETH/USD,3452.7,3521.7,3382.7,3392.1,5679414.311662269,1674.30627389
1721088000000,2024-07-16 00:00:00,ETH/USD,3491.2,3503.5,3355.2,3448.2,8783897.646483036,2547.38635998
1721001600000,2024-07-15 00:00:00,ETH/USD,3251.9,3500.0,3241.8,3495.9,11590803.390787488,3315.54203232
1720915200000,2024-07-14 00:00:00,ETH/USD,3185.7,3275.0,3172.5,3250.2,5313269.637644219,1634.7515961
1720828800000,2024-07-13 00:00:00,ETH/USD,3143.1,3210.0,3122.4,3183.0,4537280.1741753,1425.4728791
1720742400000,2024-07-12 00:00:00,ETH/USD,3107.4,3163.3,3054.4,3142.1,8372153.572580525,2664.50895025
1720656000000,2024-07-11 00:00:00,ETH/USD,3106.5,3223.5,3062.6,3105.4,6125598.304228136,1972.56337484
1720569600000,2024-07-10 00:00:00,ETH/USD,3072.0,3158.5,3033.8,3110.4,4460283.437455872,1433.99030268
1720483200000,2024-07-09 00:00:00,ETH/USD,3027.2,3119.1,3011.8,3073.6,12842689.9312728,4178.3868855
1720396800000,2024-07-08 00:00:00,ETH/USD,2940.6,3103.9,2834.7,3026.2,11201153.079604322,3701.39220131
1720310400000,2024-07-07 00:00:00,ETH/USD,3067.2,3079.2,2932.0,2940.5,21335672.070360057,7255.79733731
1720224000000,2024-07-06 00:00:00,ETH/USD,2987.0,3085.1,2962.7,3072.6,9243148.909564584,3008.24998684
1720137600000,2024-07-05 00:00:00,ETH/USD,3062.9,3111.3,2815.8,2989.3,45554973.27716776,15239.344755350001
1720051200000,2024-07-04 00:00:00,ETH/USD,3299.1,3315.8,3055.8,3060.0,23460967.4185206,7666.98281651
1719964800000,2024-07-03 00:00:00,ETH/USD,3425.1,3435.6,3255.2,3299.9,15294018.340736246,4634.69145754
1719878400000,2024-07-02 00:00:00,ETH/USD,3445.8,3466.0,3405.7,3425.3,4305599.123037517,1256.99913089
1719792000000,2024-07-01 00:00:00,ETH/USD,3441.7,3526.7,3428.2,3444.8,6903883.269557617,2004.1463276700001
1719705600000,2024-06-30 00:00:00,ETH/USD,3381.2,3462.6,3358.5,3441.0,3600819.19566849,1046.44556689
1719619200000,2024-06-29 00:00:00,ETH/USD,3386.4,3412.4,3377.4,3381.2,1767400.221334444,522.71389487
1719532800000,2024-06-28 00:00:00,ETH/USD,3455.3,3492.5,3371.5,3384.6,4829065.10614275,1426.77572125
1719446400000,2024-06-27 00:00:00,ETH/USD,3376.2,3482.0,3365.2,3452.4,9732595.483720368,2819.08106932
1719360000000,2024-06-26 00:00:00,ETH/USD,3397.7,3428.9,3331.0,3374.8,12990877.724597918,3849.37706667
1719273600000,2024-06-25 00:00:00,ETH/USD,3356.4,3431.9,3340.8,3399.2,14918664.130601728,4388.87506784
1719187200000,2024-06-24 00:00:00,ETH/USD,3425.1,3437.4,3231.8,3356.5,22827521.871157326,6800.98968305
1719100800000,2024-06-23 00:00:00,ETH/USD,3505.0,3525.0,3412.4,3427.7,5265153.850103078,1536.06028827
1719014400000,2024-06-22 00:00:00,ETH/USD,3526.7,3526.7,3481.7,3500.5,2183644.837499685,623.80940937
1718928000000,2024-06-21 00:00:00,ETH/USD,3522.8,3551.5,3455.6,3523.8,8611072.64129637,2443.68938115
1718841600000,2024-06-20 00:00:00,ETH/USD,3557.9,3628.6,3492.2,3519.7,8870160.958718063,2520.14687579
1718755200000,2024-06-19 00:00:00,ETH/USD,3488.4,3591.7,3471.6,3560.6,9730642.808564931,2732.86603622
1718668800000,2024-06-18 00:00:00,ETH/USD,3517.3,3517.6,3354.1,3487.2,12697169.409834526,3641.07863324
1718582400000,2024-06-17 00:00:00,ETH/USD,3625.7,3637.8,3468.5,3515.3,13508429.252742631,3842.75289527
1718496000000,2024-06-16 00:00:00,ETH/USD,3568.1,3652.5,3540.9,3624.2,3981638.5683768457,1098.62550863
1718409600000,2024-06-15 00:00:00,ETH/USD,3481.3,3593.6,3474.1,3566.8,7106829.67097578,1992.49458085
1718323200000,2024-06-14 00:00:00,ETH/USD,3470.8,3530.8,3365.0,3479.4,9865348.318826659,2835.35906157
1718236800000,2024-06-13 00:00:00,ETH/USD,3559.0,3562.0,3428.4,3470.3,7448951.325962605,2146.48627668
1718150400000,2024-06-12 00:00:00,ETH/USD,3498.7,3655.3,3463.6,3559.8,18710368.277494382,5256.0167081
1718064000000,2024-06-11 00:00:00,ETH/USD,3666.6,3671.5,3431.9,3499.6,42478858.602808215,12138.20396697
1717977600000,2024-06-10 00:00:00,ETH/USD,3706.2,3712.3,3645.6,3666.8,6145997.05782546,1676.12006595
1717891200000,2024-06-09 00:00:00,ETH/USD,3681.8,3721.8,3666.8,3705.8,3874896.499960024,1045.63022828
1717804800000,2024-06-08 00:00:00,ETH/USD,3678.6,3711.7,3663.2,3682.5,1773304.6101691502,481.54911342
1717718400000,2024-06-07 00:00:00,ETH/USD,3814.0,3842.0,3584.4,3682.0,19334526.42525236,5251.09354298
1717632000000,2024-06-06 00:00:00,ETH/USD,3861.6,3878.7,3761.1,3815.6,6079811.304164312,1593.40898002
1717545600000,2024-06-05 00:00:00,ETH/USD,3812.9,3884.0,3781.7,3864.5,14630987.391500426,3785.99751365
1717459200000,2024-06-04 00:00:00,ETH/USD,3768.3,3832.1,3731.9,3815.1,21034656.529830307,5513.52691406
1717372800000,2024-06-03 00:00:00,ETH/USD,3781.9,3852.1,3761.6,3767.2,12560767.811429048,3334.24501259
1717286400000,2024-06-02 00:00:00,ETH/USD,3817.6,3837.9,3754.9,3783.4,3107649.91564268,821.3907902
1717200000000,2024-06-01 00:00:00,ETH/USD,3768.2,3833.7,3755.3,3817.4,3067721.871389184,803.61551616"""

# Convert the data into DataFrames
bitcoin_df = pd.read_csv(io.StringIO(bitcoin_data))
ethereum_df = pd.read_csv(io.StringIO(ethereum_data))

# Convert 'date' column to datetime type for better plotting
bitcoin_df['date'] = pd.to_datetime(bitcoin_df['date'])
ethereum_df['date'] = pd.to_datetime(ethereum_df['date'])

# Plotting Bitcoin closing prices
plt.figure(figsize=(10,6))
plt.plot(bitcoin_df['date'], bitcoin_df['close'], label="Bitcoin", color="orange")
plt.title("Bitcoin Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Plotting Ethereum closing prices
plt.figure(figsize=(10,6))
plt.plot(ethereum_df['date'], ethereum_df['close'], label="Ethereum", color="blue")
plt.title("Ethereum Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Function to plot the original and transformed data
def plot_wavelet_results(original, cA, cD, title):
    plt.figure(figsize=(14, 8))

    # Plot original data
    plt.subplot(3, 1, 1)
    plt.plot(original, label=f'{title} Original Data')
    plt.title(f'{title} Original Data')
    plt.legend()

    # Plot DWT approximation coefficients
    plt.subplot(3, 1, 2)
    plt.plot(cA, label=f'{title} DWT Approximation Coefficients')
    plt.title(f'{title} DWT Approximation Coefficients')
    plt.legend()

    # Plot DWT detail coefficients
    plt.subplot(3, 1, 3)
    plt.plot(cD, label=f'{title} DWT Detail Coefficients')
    plt.title(f'{title} DWT Detail Coefficients')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load Bitcoin and Ethereum data into DataFrames
bitcoin_df = pd.read_csv(StringIO(bitcoin_data))
ethereum_df = pd.read_csv(StringIO(ethereum_data))

# Prepare the data by selecting the 'close' prices for both Bitcoin and Ethereum
bitcoin_close = bitcoin_df['close'].values
ethereum_close = ethereum_df['close'].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
bitcoin_close_scaled = scaler.fit_transform(bitcoin_close.reshape(-1, 1)).flatten()
ethereum_close_scaled = scaler.fit_transform(ethereum_close.reshape(-1, 1)).flatten()

# DWT (Discrete Wavelet Transform)
wavelet = 'db4'  # Daubechies wavelet of order 4
###########
# Perform DWT on the Bitcoin data
coeffs_bitcoin = pywt.wavedec(bitcoin_close_scaled, wavelet, level=2)
cA_bitcoin, cD_bitcoin = coeffs_bitcoin[0], coeffs_bitcoin[1]
bitcoin_approximation = coeffs_bitcoin[0]  # Scaling coefficients
bitcoin_details = coeffs_bitcoin [1]  # Wavelet coefficients

# Perform DWT on the Ethereum data
coeffs_ethereum = pywt.wavedec(ethereum_close_scaled, wavelet, level=2)
cA_ethereum, cD_ethereum = coeffs_ethereum[0], coeffs_ethereum[1]
ethereum_approximation = coeffs_ethereum[0]  # Scaling coefficients
ethereum_details = coeffs_ethereum[1]  # Wavelet coefficients

# Plot original vs DWT-transformed data for Bitcoin and Ethereum
plot_wavelet_results(bitcoin_close_scaled, cA_bitcoin, cD_bitcoin, "Bitcoin")
plot_wavelet_results(ethereum_close_scaled, cA_ethereum, cD_ethereum, "Ethereum")
##############
# Plot Scaling and Wavelet functions for Bitcoin
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(bitcoin_approximation, label="Scaling Function (Bitcoin)", color='blue')
plt.title("Bitcoin: Scaling Function")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(bitcoin_details, label="Wavelet Function (Bitcoin)", color='green')
plt.title("Bitcoin: Wavelet Function")
plt.legend()
plt.tight_layout()
plt.show()

# Plot Scaling and Wavelet functions for Ethereum
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ethereum_approximation, label="Scaling Function (Ethereum)", color='red')
plt.title("Ethereum: Scaling Function")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(ethereum_details, label="Wavelet Function (Ethereum)", color='orange')
plt.title("Ethereum: Wavelet Function")
plt.legend()
plt.tight_layout()
plt.show()

# Continuous Wavelet Transform (CWT)
widths = np.arange(1, 128)

# Perform CWT on Bitcoin data using a Morlet wavelet
cwt_bitcoin = cwt(bitcoin_close_scaled, morlet, widths)

# Perform CWT on Ethereum data using a Morlet wavelet
cwt_ethereum = cwt(ethereum_close_scaled, morlet, widths)

# Plotting the CWT results
plt.figure(figsize=(14, 8))

# Plot CWT for Bitcoin (magnitude of complex numbers)
plt.subplot(2, 1, 1)
plt.imshow(np.abs(cwt_bitcoin), extent=[0, len(bitcoin_close_scaled), 1, 128], cmap='PRGn', aspect='auto', vmax=abs(cwt_bitcoin).max(), vmin=-abs(cwt_bitcoin).max())
plt.title('Bitcoin Continuous Wavelet Transform (CWT) Magnitude')
plt.colorbar()

# Plot CWT for Ethereum (magnitude of complex numbers)
plt.subplot(2, 1, 2)
plt.imshow(np.abs(cwt_ethereum), extent=[0, len(ethereum_close_scaled), 1, 128], cmap='PRGn', aspect='auto', vmax=abs(cwt_ethereum).max(), vmin=-abs(cwt_ethereum).max())
plt.title('Ethereum Continuous Wavelet Transform (CWT) Magnitude')
plt.colorbar()

plt.tight_layout()
plt.show()

#Bandpass Filter
def apply_wavelet_bandpass_filter(coeffs, low_level, high_level):
    filtered_coeffs = coeffs.copy()
    for i in range(len(filtered_coeffs)):
        if i < low_level or i > high_level:
            filtered_coeffs[i] = np.zeros_like(filtered_coeffs[i])  # Zero out unwanted coefficients
    return filtered_coeffs

# Apply bandpass filter to Bitcoin and Ethereum
filtered_coeffs_bitcoin = apply_wavelet_bandpass_filter(coeffs_bitcoin, 1, 2)  # Keep levels 1 and 2 (bandpass)
filtered_coeffs_ethereum = apply_wavelet_bandpass_filter(coeffs_ethereum, 1, 2)  # Keep levels 1 and 2 (bandpass)

# Reconstruct the filtered signal
filtered_bitcoin = pywt.waverec(filtered_coeffs_bitcoin, wavelet)
filtered_ethereum = pywt.waverec(filtered_coeffs_ethereum, wavelet)

# Plot the original and bandpass filtered signals
plt.figure(figsize=(14, 8))

# Plot original Bitcoin data
plt.subplot(2, 1, 1)
plt.plot(bitcoin_close_scaled, label='Original Bitcoin Data')
plt.plot(filtered_bitcoin, label='Bandpass Filtered Bitcoin Data', linestyle='--')
plt.title('Bitcoin: Original vs Bandpass Filtered')
plt.legend()

# Plot original Ethereum data
plt.subplot(2, 1, 2)
plt.plot(ethereum_close_scaled, label='Original Ethereum Data')
plt.plot(filtered_ethereum, label='Bandpass Filtered Ethereum Data', linestyle='--')
plt.title('Ethereum: Original vs Bandpass Filtered')
plt.legend()

plt.tight_layout()
plt.show()

# Define function to create sequences of inputs and outputs
def create_sequences(data, window_size):
    X, Y = [], []
    for i in range(1, len(data) - window_size - 1, 1):
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append(data[i + j])
        temp2.append(data[i + window_size])
        X.append(np.array(temp).reshape(100, 1))
        Y.append(np.array(temp2).reshape(1, 1))
    return np.array(X), np.array(Y)

# Define function to process the data, train, and evaluate model for Bitcoin/Ethereum
def process_and_train(df, crypto_name):
    # Normalize the 'close' column
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['close']])

    # Create sequences of inputs and outputs
    window_size = 100
    X, Y = create_sequences(df_scaled, window_size)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # Reshape the data for the CNN-LSTM model
    train_X = x_train.reshape(x_train.shape[0], 1, 100, 1)
    test_X = x_test.reshape(x_test.shape[0], 1, 100, 1)

    # Create the Sequential model
    model = tf.keras.Sequential()

    # CNN layers
    model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.5))

    # Final dense layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse', 'mae']
    )

    # Train the model
    history = model.fit(train_X, np.array(y_train), validation_data=(test_X, np.array(y_test)), epochs=40, batch_size=40, verbose=1, shuffle=True)

    # Plot the loss, mse, and mae over epochs
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{crypto_name} - Loss Over Epochs')
    plt.show()

    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title(f'{crypto_name} - MSE Over Epochs')
    plt.show()

    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title(f'{crypto_name} - MAE Over Epochs')
    plt.show()

    # Evaluate the model
    print(f"Evaluating {crypto_name} model...")
    model.evaluate(test_X, np.array(y_test))

    # Predict the test set
    yhat_probs = model.predict(test_X, verbose=0)
    yhat_probs = yhat_probs[:, 0]  # Flatten to 1D array

    # Calculate explained variance, R2 score, and max error
    var = explained_variance_score(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'Explained Variance ({crypto_name}): {var}')

    r2 = r2_score(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'R2 Score ({crypto_name}): {r2}')

    max_err = max_error(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'Max Error ({crypto_name}): {max_err}')

    # Inverse transform the predictions and true values for comparison
    predicted = scaler.inverse_transform(model.predict(test_X))
    test_label = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    predicted = np.array(predicted[:, 0]).reshape(-1, 1)

    # Plot the real vs predicted stock prices
    plt.plot(predicted, color='green', label='Predicted Stock Price')
    plt.plot(test_label, color='red', label='Real Stock Price')
    plt.title(f'{crypto_name} Stock Price Prediction')
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Save the model
    model.save(f'{crypto_name}_model.h5')
    return model

# Process and train for both Bitcoin and Ethereum

# For Bitcoin
process_and_train(bitcoin_df, "Bitcoin")

# For Ethereum
process_and_train(ethereum_df, "Ethereum")

#################

def prepare_and_predict(data, crypto_name, days_to_predict, window_size=30):
    """
    Prepare data, train the model, and make predictions.

    Parameters:
    - data: The historical price data for the cryptocurrency.
    - crypto_name: Name of the cryptocurrency.
    - days_to_predict: Number of days to predict into the future.
    - window_size: Size of the sliding window for LSTM input.
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(data.values.reshape(-1, 1))

    # Prepare training data
    X_train, y_train = [], []
    for i in range(window_size, len(scaled_prices)):
        X_train.append(scaled_prices[i - window_size:i, 0])
        y_train.append(scaled_prices[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

    # Predict on training data
    predicted_train = model.predict(X_train)
    predicted_train = scaler.inverse_transform(predicted_train)
    real_train = scaler.inverse_transform(scaled_prices[window_size:])

    # Forecast future prices
    future_input = scaled_prices[-window_size:].reshape(1, window_size, 1)
    future_predictions = []
    for _ in range(days_to_predict):
        next_prediction = model.predict(future_input)[0]
        future_predictions.append(next_prediction)
        future_input = np.append(future_input[:, 1:, :], [[next_prediction]], axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(predicted_train)), predicted_train, color='green', label=f'{crypto_name} Predicted Training Data')
    plt.plot(range(len(real_train)), real_train, color='red', label=f'{crypto_name} Real Training Data')
    plt.plot(range(len(real_train), len(real_train) + days_to_predict), future_predictions, color='blue', label=f'{crypto_name} Future Predictions')
    plt.title(f'{crypto_name} Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#Accuracy
# Step 1: Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the images to a range of [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Optionally split off some of the training set to use for validation
X_valid = X_train[:5000]  # First 5000 samples for validation
y_valid = y_train[:5000]

X_train = X_train[5000:]  # Remaining samples for training
y_train = y_train[5000:]

# Step 2: Define the model architecture
model = models.Sequential()

model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# Example layers for a basic Convolutional Neural Network (CNN)

# LSTM layers
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))


# Step 3: Compile the model with accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use for integer labels
              metrics=['accuracy'])

# Step 4: Print the model summary
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, 
                    epochs=1,  # Change this for more epochs
                    validation_data=(X_valid, y_valid))

# Step 6: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

############################################
