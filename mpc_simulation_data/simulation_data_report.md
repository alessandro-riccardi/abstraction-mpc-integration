###########################################################
-----------------------------------------------------------
###########################################################
# DUBINS CAR
Sim 00
'abstraction_data_Dubins_small_03'
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
rollout_target horizon 3


MPC Montecarlo simulation time: 7759.670603 seconds
Average MPC simulation time: 77.596706 seconds
Expected MPC step computation time: 5.204339773775972
Expected cost associated to the policy: 511.23380274565227
Expected cost associated to the policy state: 386.44908701223716
Expected cost associated to the policy input: 124.78471573341528
Expected cost associated to the MPC: 540.9959575396914
Expected cost associated to the MPC state: 425.28289574291654
Expected cost associated to the MPC input: 115.71306179677478
MPC performance improvement w.r.t. policy: -0.05821632809528121
MPC performance improvement w.r.t. policy state: -0.10048880961504159
MPC performance improvement w.r.t. policy input: 0.0726984381326059
Expected path length policy: 54.61873104662978
Expected path length MPC: 50.21372363964629
MPC performance improvement in trajectory length w.r.t. policy: 0.08065012354869239
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 01
'abstraction_data_Dubins_small_03'
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
fixed target rolluout

MPC Montecarlo simulation time: 804.744443 seconds
Average MPC simulation time: 80.474444 seconds
Expected MPC step computation time: 5.061285806918273
Expected cost associated to the policy: 2121.6304286693276
Expected cost associated to the policy state: 1997.797504517133
Expected cost associated to the policy input: 123.83292415219478
Expected cost associated to the MPC: 2521.3110793343017
Expected cost associated to the MPC state: 2401.156335872292
Expected cost associated to the MPC input: 120.15474346200945
MPC performance improvement w.r.t. policy: -0.1883837285062182
MPC performance improvement w.r.t. policy state: -0.20190175953425804
MPC performance improvement w.r.t. policy input: 0.029702768592177686
Expected path length policy: 50.283127070149284
Expected path length MPC: 56.11251236387081
MPC performance improvement in trajectory length w.r.t. policy: -0.11593124042562097
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 02
'abstraction_data_Dubins_small_02'
self.epsilons = np.array([0.1,0.2])
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
closest target cell
100 simuated traces

MPC Montecarlo simulation time: 4472.776684 seconds
Average MPC simulation time: 44.727767 seconds
Expected MPC step computation time: 3.0038795728676844
Expected cost associated to the policy: 1653.371146679939
Expected cost associated to the policy state: 1528.5864309465244
Expected cost associated to the policy input: 124.78471573341528
Expected cost associated to the MPC: 1771.6725710157255
Expected cost associated to the MPC state: 1648.528011511881
Expected cost associated to the MPC input: 123.14455950384472
MPC performance improvement w.r.t. policy: -0.07155164439233275
MPC performance improvement w.r.t. policy state: -0.07846568446318525
MPC performance improvement w.r.t. policy input: 0.013143887213515226
Expected path length policy: 54.61873104662978
Expected path length MPC: 53.19687642294536
MPC performance improvement in trajectory length w.r.t. policy: 0.026032362825685995
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 03
'abstraction_data_Dubins_small_03'
self.epsilons = np.array([0.15,0.3])
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
closest target cell
100 simuated traces

MPC Montecarlo simulation time: 5030.518388 seconds
Average MPC simulation time: 50.305184 seconds
Expected MPC step computation time: 3.313911981225311
Expected cost associated to the policy: 1653.371146679939
Expected cost associated to the policy state: 1528.5864309465244
Expected cost associated to the policy input: 124.78471573341528
Expected cost associated to the MPC: 1805.4129083992004
Expected cost associated to the MPC state: 1689.6569581221597
Expected cost associated to the MPC input: 115.75595027704108
MPC performance improvement w.r.t. policy: -0.09195863979153729
MPC performance improvement w.r.t. policy state: -0.10537220788745194
MPC performance improvement w.r.t. policy input: 0.07235473834522221
Expected path length policy: 54.61873104662978
Expected path length MPC: 50.934635743204346
MPC performance improvement in trajectory length w.r.t. policy: 0.06745113320703489
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 04
'abstraction_data_Dubins_small_06'
self.epsilons = np.array([0.17,0.34])
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
closest target cell
100 simuated traces
MPC Montecarlo simulation time: 4735.896891 seconds
Average MPC simulation time: 47.358969 seconds
Expected MPC step computation time: 3.167824007491667
Expected cost associated to the policy: 1653.371146679939
Expected cost associated to the policy state: 1528.5864309465244
Expected cost associated to the policy input: 124.78471573341528
Expected cost associated to the MPC: 1756.8744986694855
Expected cost associated to the MPC state: 1639.3829760483623
Expected cost associated to the MPC input: 117.49152262112283
MPC performance improvement w.r.t. policy: -0.06260140210949428
MPC performance improvement w.r.t. policy state: -0.07248300970016519
MPC performance improvement w.r.t. policy input: 0.058446205285856645
Expected path length policy: 54.61873104662978
Expected path length MPC: 46.25756971030989
MPC performance improvement in trajectory length w.r.t. policy: 0.1530823066757392
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0


Sim 05
REDUCED NOISE
'abstraction_data_Dubins_small_03'
self.epsilons = np.array([0.15,0.3])
'--cov_noise', '[0.01]',
'--cov_initial_state', '0.1'
closest target cell
100 simuated traces

Sim 06
'abstraction_data_Dubins_small_03'
self.epsilons = np.array([0.15,0.3])
'--cov_noise', '[0.1]',
'--cov_initial_state', '0.1'
closest target according to nominal trajectory
100 simuated traces



###########################################################
-----------------------------------------------------------
###########################################################

# MOUNTAIN CAR
Sim 00
abstraction_data_MountainCar_02
epsilon = 0.1
ROULLOUT TARGET
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 27467.557853 seconds
Average MPC simulation time: 274.675579 seconds
Expected MPC step computation time: 5.103596776885906
Expected cost associated to the policy: 59.21177766857999
Expected cost associated to the policy state: 16.594277668580006
Expected cost associated to the policy input: 42.6175
Expected cost associated to the MPC: 55.826215881725076
Expected cost associated to the MPC state: 18.996193484614885
Expected cost associated to the MPC input: 36.830022397110206
MPC performance improvement w.r.t. policy: 0.05717716846477017
MPC performance improvement w.r.t. policy state: -0.14474361969865823
MPC performance improvement w.r.t. policy input: 0.13580049516958514
Expected path length policy: 3.124324526375741
Expected path length MPC: 3.3266875882141864
MPC performance improvement in trajectory length w.r.t. policy: -0.06477017996372791
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.02464661109097499

Sim 01
abstraction_data_MountainCar_02
epsilon = 0.1
FIXED TARGET
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 27258.793214 seconds
Average MPC simulation time: 272.587932 seconds
Expected MPC step computation time: 5.061985740742805
Expected cost associated to the policy: 94.35853026046706
Expected cost associated to the policy state: 51.74103026046707
Expected cost associated to the policy input: 42.6175
Expected cost associated to the MPC: 94.3260128735466
Expected cost associated to the MPC state: 57.44248066353686
Expected cost associated to the MPC input: 36.88353221000972
MPC performance improvement w.r.t. policy: 0.0003446152333095309
MPC performance improvement w.r.t. policy state: -0.11019205405011046
MPC performance improvement w.r.t. policy input: 0.13454491206641128
Expected path length policy: 3.124324526375741
Expected path length MPC: 3.3180051820470005
MPC performance improvement in trajectory length w.r.t. policy: -0.06199120931138724
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.023217848721204425


Sim 03
abstraction_data_MountainCar_02
epsilon = 0.1
Rollout TARGET horizon 105
10 simulations
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 1542.931409 seconds
Average MPC simulation time: 154.293141 seconds
Expected MPC step computation time: 2.894805645966255
Expected cost associated to the policy: 95.4569122874627
Expected cost associated to the policy state: 53.33191228746269
Expected cost associated to the policy input: 42.125
Expected cost associated to the MPC: 95.6362483634592
Expected cost associated to the MPC state: 59.19909329607957
Expected cost associated to the MPC input: 36.437155067379635
MPC performance improvement w.r.t. policy: -0.001878712307982866
MPC performance improvement w.r.t. policy state: -0.11001257515373464
MPC performance improvement w.r.t. policy input: 0.13502302510671493
Expected path length policy: 3.0889354521704937
Expected path length MPC: 3.3114800709285612
MPC performance improvement in trajectory length w.r.t. policy: -0.0720457329730515
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.023809523809523808

Sim 04
abstraction_data_MountainCar_02
epsilon = 0.1
Fixed Target
10 simulations
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 1532.055101 seconds
Average MPC simulation time: 153.205510 seconds
Expected MPC step computation time: 2.8798028208646675
Expected cost associated to the policy: 92.78519958969846
Expected cost associated to the policy state: 50.66019958969845
Expected cost associated to the policy input: 42.125
Expected cost associated to the MPC: 92.76975326420072
Expected cost associated to the MPC state: 56.156314683120875
Expected cost associated to the MPC input: 36.61343858107984
MPC performance improvement w.r.t. policy: 0.00016647402350866702
MPC performance improvement w.r.t. policy state: -0.10848980339469572
MPC performance improvement w.r.t. policy input: 0.13083825326813436
Expected path length policy: 3.0889354521704937
Expected path length MPC: 3.3143722412369754
MPC performance improvement in trajectory length w.r.t. policy: -0.07298203298747286
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.02564102564102564


Sim 05
abstraction_data_MountainCar_02
epsilon = 0.1
Fixed Target
100 simulations
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 1532.055101 seconds
Average MPC simulation time: 153.205510 seconds
Expected MPC step computation time: 2.8798028208646675
Expected cost associated to the policy: 92.78519958969846
Expected cost associated to the policy state: 50.66019958969845
Expected cost associated to the policy input: 42.125
Expected cost associated to the MPC: 92.76975326420072
Expected cost associated to the MPC state: 56.156314683120875
Expected cost associated to the MPC input: 36.61343858107984
MPC performance improvement w.r.t. policy: 0.00016647402350866702
MPC performance improvement w.r.t. policy state: -0.10848980339469572
MPC performance improvement w.r.t. policy input: 0.13083825326813436
Expected path length policy: 3.0889354521704937
Expected path length MPC: 3.3143722412369754
MPC performance improvement in trajectory length w.r.t. policy: -0.07298203298747286
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.02564102564102564


Sim 06
abstraction_data_MountainCar_04
epsilon = 0.15
Fixed Target
100 simulations
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'

MPC Montecarlo simulation time: 16054.199587 seconds
Average MPC simulation time: 160.541996 seconds
Expected MPC step computation time: 2.87916061459828
Expected cost associated to the policy: 94.35853026046706
Expected cost associated to the policy state: 51.74103026046707
Expected cost associated to the policy input: 42.6175
Expected cost associated to the MPC: 94.16314541828322
Expected cost associated to the MPC state: 59.800311362784086
Expected cost associated to the MPC input: 34.362834055499086
MPC performance improvement w.r.t. policy: 0.002070664323029381
MPC performance improvement w.r.t. policy state: -0.15576189847295607
MPC performance improvement w.r.t. policy input: 0.19369193276238428
Expected path length policy: 3.124324526375741
Expected path length MPC: 3.3470518033192906
MPC performance improvement in trajectory length w.r.t. policy: -0.0712881376640846
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.02261174408413672


Sim 07
abstraction_data_MountainCar_03
epsilon = 0.2
Fixed Target
100 simulations
'--cov_noise', '[0.005, 0.0005]',
'--cov_initial_state', '0.0025'
MPC Montecarlo simulation time: 17001.281630 seconds
Average MPC simulation time: 170.012816 seconds
Expected MPC step computation time: 3.052294727091559
Expected cost associated to the policy: 94.35853026046706
Expected cost associated to the policy state: 51.74103026046707
Expected cost associated to the policy input: 42.6175
Expected cost associated to the MPC: 91.48018925955579
Expected cost associated to the MPC state: 60.47178118373682
Expected cost associated to the MPC input: 31.008408075818988
MPC performance improvement w.r.t. policy: 0.030504300914457994
MPC performance improvement w.r.t. policy state: -0.16873941008361637
MPC performance improvement w.r.t. policy input: 0.2724019927067757
Expected path length policy: 3.124324526375741
Expected path length MPC: 3.2949844911034245
MPC performance improvement in trajectory length w.r.t. policy: -0.05462299555854777
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.021777309448542326

###########################################################
-----------------------------------------------------------
###########################################################

# DOUBLE INTEGRATOR PREDICTION HORIZON 3


Sim 01
abstraction_data_DoubleIntegrator_02.pkl
epsilon = 0.1

MPC Montecarlo simulation time: 57.670468 seconds
Average MPC simulation time: 0.576705 seconds
Expected MPC step computation time: 0.12758953097344383

Expected cost associated to the policy: 142.7132177208478
Expected cost associated to the policy state: 75.19821772084784
Expected cost associated to the policy input: 67.515
Expected cost associated to the MPC: 139.02956199991272
Expected cost associated to the MPC state: 74.08251776724383
Expected cost associated to the MPC input: 64.94704423266887
MPC performance improvement w.r.t. policy: 0.02581159460744875
MPC performance improvement w.r.t. policy state: 0.014836787192825884
MPC performance improvement w.r.t. policy input: 0.03803533684856886

Expected path length policy: 21.436671896990482
Expected path length MPC: 21.12314036318063
MPC performance improvement in trajectory length w.r.t. policy: 0.014625942651754098

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 02
abstraction_data_DoubleIntegrator_04.pkl
epsilon = 0.3

MPC Montecarlo simulation time: 59.689031 seconds
Average MPC simulation time: 0.596890 seconds
Expected MPC step computation time: 0.12836350752688538

Expected cost associated to the policy: 142.7132177208478
Expected cost associated to the policy state: 75.19821772084784
Expected cost associated to the policy input: 67.515
Expected cost associated to the MPC: 136.32888602555
Expected cost associated to the MPC state: 74.19670462075557
Expected cost associated to the MPC input: 62.132181404794416
MPC performance improvement w.r.t. policy: 0.0447353916985167
MPC performance improvement w.r.t. policy state: 0.013318309003148232
MPC performance improvement w.r.t. policy input: 0.07972774339340272

Expected path length policy: 21.436671896990482
Expected path length MPC: 20.992331575099374
MPC performance improvement in trajectory length w.r.t. policy: 0.020728046033745118

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0


Sim 03
abstraction_data_DoubleIntegrator_06
epsilon = 0.5

MPC Montecarlo simulation time: 71.396821 seconds
Average MPC simulation time: 0.713968 seconds
Expected MPC step computation time: 0.15288398565311428

Expected cost associated to the policy: 142.7132177208478
Expected cost associated to the policy state: 75.19821772084784
Expected cost associated to the policy input: 67.515
Expected cost associated to the MPC: 126.78204979467073
Expected cost associated to the MPC state: 71.48773805976495
Expected cost associated to the MPC input: 55.29431173490579
MPC performance improvement w.r.t. policy: 0.11163064066945094
MPC performance improvement w.r.t. policy state: 0.04934265430142237
MPC performance improvement w.r.t. policy input: 0.1810070097770008

Expected path length policy: 21.436671896990482
Expected path length MPC: 19.743405009762167
MPC performance improvement in trajectory length w.r.t. policy: 0.07898926173638154

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0


###########################################################
-----------------------------------------------------------
###########################################################

# DOUBLE INTEGRATOR PREDICTION HORIZON 5

Sim 04
abstraction_data_DoubleIntegrator_02
epsilon = 0.1

MPC Montecarlo simulation time: 145.182193 seconds
Average MPC simulation time: 1.451822 seconds
Expected MPC step computation time: 0.31492883579176323

Expected cost associated to the policy: 141.91689889092007
Expected cost associated to the policy state: 75.18189889092008
Expected cost associated to the policy input: 66.735
Expected cost associated to the MPC: 139.8637290432713
Expected cost associated to the MPC state: 74.43372952287328
Expected cost associated to the MPC input: 65.42999952039806
MPC performance improvement w.r.t. policy: 0.014467409192945202
MPC performance improvement w.r.t. policy state: 0.009951456122866853
MPC performance improvement w.r.t. policy input: 0.019554963356588684

Expected path length policy: 21.489872626628063
Expected path length MPC: 21.255731857148877
MPC performance improvement in trajectory length w.r.t. policy: 0.010895400524108423

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

Sim 05
abstraction_data_DoubleIntegrator_04
epsilon = 0.3

MPC Montecarlo simulation time: 231.274809 seconds
Average MPC simulation time: 2.312748 seconds
Expected MPC step computation time: 0.49207406255318736

Expected cost associated to the policy: 141.91689889092007
Expected cost associated to the policy state: 75.18189889092008
Expected cost associated to the policy input: 66.735
Expected cost associated to the MPC: 136.06166884265636
Expected cost associated to the MPC state: 73.95471272676642
Expected cost associated to the MPC input: 62.1069561158899
MPC performance improvement w.r.t. policy: 0.041258159486448065
MPC performance improvement w.r.t. policy state: 0.016322893971249197
MPC performance improvement w.r.t. policy input: 0.06934957494733052

Expected path length policy: 21.489872626628063
Expected path length MPC: 20.97698628695371
MPC performance improvement in trajectory length w.r.t. policy: 0.023866420643128272

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0


Sim 06
abstraction_data_DoubleIntegrator_06
epsilon = 0.5

MPC Montecarlo simulation time: 269.842400 seconds
Average MPC simulation time: 2.698424 seconds
Expected MPC step computation time: 0.5741327663829694

Expected cost associated to the policy: 141.91689889092007
Expected cost associated to the policy state: 75.18189889092008
Expected cost associated to the policy input: 66.735
Expected cost associated to the MPC: 127.0079082451881
Expected cost associated to the MPC state: 70.94382338979268
Expected cost associated to the MPC input: 56.064084855395386
MPC performance improvement w.r.t. policy: 0.10505437169389731
MPC performance improvement w.r.t. policy state: 0.056370955823772606
MPC performance improvement w.r.t. policy input: 0.1598998298434796

Expected path length policy: 21.489872626628063
Expected path length MPC: 19.742015990110747
MPC performance improvement in trajectory length w.r.t. policy: 0.08133396911583135

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0



###########################################################
-----------------------------------------------------------
###########################################################

# MOUNTAIN CAR

SIM_ID 04
epsilon = 0.15
Final simulation 100 traces conventional cost function
MPC Montecarlo simulation time: 27641.002997 seconds
Average MPC simulation time: 276.410030 seconds
Expected MPC step computation time: 4.861238655786142
Expected cost associated to the nominal policy: 192.80233934260627
Expected cost associated to the nominal policy state: 107.41983934260628
Expected cost associated to the nominal policy input: 85.3825
Expected cost associated to the policy: 99.1715451500751
Expected cost associated to the policy state: 56.214045150075165
Expected cost associated to the policy input: 42.9575
Expected cost associated to the MPC: 91.82257163123163
Expected cost associated to the MPC state: 59.02418022331409
Expected cost associated to the MPC input: 32.79839140791756
MPC performance improvement w.r.t. policy: 0.07410365047476442
MPC performance improvement w.r.t. policy state: -0.04998991027485533
MPC performance improvement w.r.t. policy input: 0.23649208152435408
MPC performance improvement w.r.t. nominal policy: 0.5237476270032981
MPC performance improvement w.r.t. noinal policy state: 0.4505281279088347
MPC performance improvement w.r.t. noinal policy input: 0.615865178368898
Expected path length policy: 3.381617008218346
Expected path length nominal policy: 3.109645664647471
Expected path length MPC: 3.333550820125189
MPC performance improvement in trajectory length w.r.t. policy: 0.014213965678650742
MPC performance improvement in trajectory length w.r.t. nominal policy: -0.07200343049474145
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.002106002106002106


###########################################################

SIM_ID 02
epsilon = 0.1
MPC Montecarlo simulation time: 39507.542182 seconds
Average MPC simulation time: 395.075422 seconds
Expected MPC step computation time: 7.046110608560708
Expected cost associated to the nominal policy: 190.8175278982463
Expected cost associated to the nominal policy state: 106.3175278982463
Expected cost associated to the nominal policy input: 84.5
Expected cost associated to the policy: 97.1867337057151
Expected cost associated to the policy state: 55.11173370571512
Expected cost associated to the policy input: 42.075
Expected cost associated to the MPC: 90.96194179549744
Expected cost associated to the MPC state: 58.02492139435967
Expected cost associated to the MPC input: 32.937020401137794
MPC performance improvement w.r.t. policy: 0.06404981084215211
MPC performance improvement w.r.t. policy state: -0.05285966331961811
MPC performance improvement w.r.t. policy input: 0.21718311583748562
MPC performance improvement w.r.t. nominal policy: 0.5233040549398428
MPC performance improvement w.r.t. noinal policy state: 0.4542299605583964
MPC performance improvement w.r.t. noinal policy input: 0.6102127763178959
Expected path length policy: 3.3085409312002265
Expected path length nominal policy: 3.109645664647471
Expected path length MPC: 3.315656980332491
MPC performance improvement in trajectory length w.r.t. policy: -0.0021508118775737106
MPC performance improvement in trajectory length w.r.t. nominal policy: -0.06624912864738719
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0010689470871191875

###########################################################

SIM_ID 03
epsilon = 0.2
MPC Montecarlo simulation time: 40309.496413 seconds
Average MPC simulation time: 403.094964 seconds
Expected MPC step computation time: 7.068121412081328
Expected cost associated to the nominal policy: 193.57301570287828
Expected cost associated to the nominal policy state: 107.6930157028783
Expected cost associated to the nominal policy input: 85.88
Expected cost associated to the policy: 99.9422215103471
Expected cost associated to the policy state: 56.48722151034718
Expected cost associated to the policy input: 43.455
Expected cost associated to the MPC: 92.38117123890862
Expected cost associated to the MPC state: 59.14165547390017
Expected cost associated to the MPC input: 33.23951576500843
MPC performance improvement w.r.t. policy: 0.07565421457692612
MPC performance improvement w.r.t. policy state: -0.046991760128735516
MPC performance improvement w.r.t. policy input: 0.23508190622463632
MPC performance improvement w.r.t. nominal policy: 0.5227580099247533
MPC performance improvement w.r.t. noinal policy state: 0.4508310953324014
MPC performance improvement w.r.t. noinal policy input: 0.6129539384605447
Expected path length policy: 3.425884586820356
Expected path length nominal policy: 3.109645664647471
Expected path length MPC: 3.3555781889603282
MPC performance improvement in trajectory length w.r.t. policy: 0.020522115114590234
MPC performance improvement in trajectory length w.r.t. nominal policy: -0.07908699280718134
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0007008936393902226



###########################################################
-----------------------------------------------------------
###########################################################

# SMALL DUBINS

SIMULATION_ID = "01"
abstraction_data_Dubins_small_03
self.epsilons = np.array([0.15,0.3])

x0 = [-9.5,-2.5,0]
Q = [1, 1, 1]
R = [1,1]
PREDICTION_HORIZON = 3
NUMBER_EXPERIMENTS_MONTECARLO = 100
NUMBER_EXPERIMENTS_MPC = 100

MPC Montecarlo simulation time: 6036.959763 seconds
Average MPC simulation time: 60.369598 seconds
Expected MPC step computation time: 3.987423885799212

Expected cost associated to the policy: 1665.8486587107943
Expected cost associated to the policy state: 1540.7722028860758
Expected cost associated to the policy input: 125.0764558247176
Expected cost associated to the MPC: 1784.5390607883005
Expected cost associated to the MPC state: 1668.1886871499344
Expected cost associated to the MPC input: 116.35037363836602
MPC performance improvement w.r.t. policy: -0.07124921069922463
MPC performance improvement w.r.t. policy state: -0.08269651024673874
MPC performance improvement w.r.t. policy input: 0.0697659853632272

Expected path length policy: 55.86590479502375
Expected path length MPC: 50.18752647329955
MPC performance improvement in trajectory length w.r.t. policy: 0.10164300287552142

Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################

SIMULATION_ID = "02"
abstraction_data_Dubins_small_02
self.epsilons = np.array([0.1,0.2])

x0 = [-9.5,-2.5,0]
Q = [1, 1, 1]
R = [1,1]
PREDICTION_HORIZON = 3
NUMBER_EXPERIMENTS_MONTECARLO = 100
NUMBER_EXPERIMENTS_MPC = 100

MPC Montecarlo simulation time: 6601.123054 seconds
Average MPC simulation time: 66.011231 seconds
Expected MPC step computation time: 4.478373849660767
Expected cost associated to the policy: 1770.679455174623
Expected cost associated to the policy state: 1642.673391259027
Expected cost associated to the policy input: 128.00606391559577
Expected cost associated to the MPC: 1743.4793657638543
Expected cost associated to the MPC state: 1622.8466539504018
Expected cost associated to the MPC input: 120.63271181345209
MPC performance improvement w.r.t. policy: 0.015361385332212096
MPC performance improvement w.r.t. policy state: 0.012069798789051456
MPC performance improvement w.r.t. policy input: 0.05760158446091668
Expected path length policy: 55.27220832892443
Expected path length MPC: 50.90284733083078
MPC performance improvement in trajectory length w.r.t. policy: 0.07905168130955831
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################

SIMULATION_ID = "03"
abstraction_data_Dubins_small_04
# self.epsilons = np.array([0.2,0.4])

x0 = [-9.5,-2.5,0]
Q = [1, 1, 1]
R = [1,1]
PREDICTION_HORIZON = 3
NUMBER_EXPERIMENTS_MONTECARLO = 100
NUMBER_EXPERIMENTS_MPC = 100

MPC Montecarlo simulation time: 8487.730929 seconds
Average MPC simulation time: 84.877309 seconds
Expected MPC step computation time: 5.291602823379095
Expected cost associated to the policy: 1770.679455174623
Expected cost associated to the policy state: 1642.673391259027
Expected cost associated to the policy input: 128.00606391559577
Expected cost associated to the MPC: 1909.1548143364423
Expected cost associated to the MPC state: 1790.4985062810456
Expected cost associated to the MPC input: 118.65630805539527
MPC performance improvement w.r.t. policy: -0.07820464554278295
MPC performance improvement w.r.t. policy state: -0.08999057013318867
MPC performance improvement w.r.t. policy input: 0.07304150736456916
Expected path length policy: 55.27220832892443
Expected path length MPC: 46.84594051552962
MPC performance improvement in trajectory length w.r.t. policy: 0.15245035557924813
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################

SIMULATION_ID = "04"
abstraction_data_Dubins_small_05
self.epsilons = np.array([0.16,0.32])

x0 = [-9.5,-2.5,0]
Q = [1, 1, 1]
R = [1,1]

MPC Montecarlo simulation time: 10434.071847 seconds
Average MPC simulation time: 104.340718 seconds
Expected MPC step computation time: 7.059588529566931
Expected cost associated to the policy: 1770.679455174623
Expected cost associated to the policy state: 1642.673391259027
Expected cost associated to the policy input: 128.00606391559577
Expected cost associated to the MPC: 1769.5409130128753
Expected cost associated to the MPC state: 1648.3505102376002
Expected cost associated to the MPC input: 121.19040277527472
MPC performance improvement w.r.t. policy: 0.0006429973298782699
MPC performance improvement w.r.t. policy state: -0.003456024191285989
MPC performance improvement w.r.t. policy input: 0.053244830220036676
Expected path length policy: 55.27220832892443
Expected path length MPC: 48.0516255696854
MPC performance improvement in trajectory length w.r.t. policy: 0.13063676986215933
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################

SIMULATION_ID = "05"
abstraction_data_Dubins_small_06
self.epsilons = np.array([0.17,0.34])

x0 = [-9.5,-2.5,0]

Q = [1, 1, 1]
R = [1,1]

MPC Montecarlo simulation time: 7075.024611 seconds
Average MPC simulation time: 70.750246 seconds
Expected MPC step computation time: 4.722980381441905
Expected cost associated to the policy: 1770.679455174623
Expected cost associated to the policy state: 1642.673391259027
Expected cost associated to the policy input: 128.00606391559577
Expected cost associated to the MPC: 1790.4732488715263
Expected cost associated to the MPC state: 1671.096352782214
Expected cost associated to the MPC input: 119.37689608931103
MPC performance improvement w.r.t. policy: -0.011178643112991438
MPC performance improvement w.r.t. policy state: -0.017302868406118277
MPC performance improvement w.r.t. policy input: 0.0674121800352725
Expected path length policy: 55.27220832892443
Expected path length MPC: 47.09815105940731
MPC performance improvement in trajectory length w.r.t. policy: 0.14788729302931736
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################

SIMULATION_ID = "06"
abstraction_data_Dubins_small_07
self.epsilons = np.array([0.18,0.36])

x0 = [-9.5,-2.5,0]
Q = [1, 1, 1]
R = [1,1]

MPC Montecarlo simulation time: 6958.994114 seconds
Average MPC simulation time: 69.589941 seconds
Expected MPC step computation time: 4.6362385833444355
Expected cost associated to the policy: 1770.679455174623
Expected cost associated to the policy state: 1642.673391259027
Expected cost associated to the policy input: 128.00606391559577
Expected cost associated to the MPC: 1755.559980461898
Expected cost associated to the MPC state: 1640.601792596034
Expected cost associated to the MPC input: 114.95818786586335
MPC performance improvement w.r.t. policy: 0.008538798294936978
MPC performance improvement w.r.t. policy state: 0.0012611141533164227
MPC performance improvement w.r.t. policy input: 0.10193170269133411
Expected path length policy: 55.27220832892443
Expected path length MPC: 43.719086292056375
MPC performance improvement in trajectory length w.r.t. policy: 0.20902226247439804
Empirical satisfaction probability MPC: 1.0
Percentage of backup policy usage: 0.0

###########################################################
-----------------------------------------------------------
###########################################################