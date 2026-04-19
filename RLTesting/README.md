Run the code with the following command:

To run the dispactcher at the DQN folder:
python .\dispatcher.py --cfg D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.sumocfg

To generate the request: 
python .\request_chain_generator.py --report D:\FYP\FYP-DRT\SmallTestingMap\connectivity_report.json --taxi D:\FYP\FYP-DRT\SmallTestingMap\map.rou.xml --output D:\FYP\FYP-DRT\SmallTestingMap\persontrips_scale.rou.xml --num-requests 200 --depart-step 25 75 200 --max-random-deviation-pct 10

To generate the request with day mode: 
python request_chain_generator.py --report D:\FYP\DQNLargeScale1\SunwaySmallMap\connectivity_report.json --output D:\FYP\DQNLargeScale1\SunwaySmallMap\self_request.rou.xml --mode day --num-requests 2000 --num-stops 80 --net D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.net.xml --day-steps 86400 --save-stops D:\FYP\DQNLargeScale1\SunwaySmallMap\stops.json --demand-very-low 0.1  --demand-morning-peak 1.0 --demand-medium 0.5 --demand-evening-peak 1.0 --demand-low-medium 0.3 --min-reachable-pickup 1000 --min-reachable-pickup 1000


To generate the taxi based on the request chain edges:
python generate_taxi.py --stops D:\FYP\DQNLargeScale1\SunwaySmallMap\stops.json --output D:\FYP\DQNLargeScale1\SunwaySmallMap\taxi_more.rou.xml --num-taxis 15 --depart 0 --vtype-id myTaxi --color yellow --person-capacity 13 --seed 42
