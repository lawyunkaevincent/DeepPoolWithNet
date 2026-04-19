Run the code with the following command:

To train the DQN model:
python train_dqn.py --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DQNLargeScale1\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir artifacts\dqn_model --episodes 40 --batch-size 64 --replay-size 20000 --warmup-transitions 100 --gamma 0.99 --lr 0.0001 --tau 0.01 --epsilon-start 0.10 --epsilon-end 0.02

python train_dqn.py --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir artifacts/D:\FYP\DQNLargeScale1\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir artifacts/dqn_model --episodes 100 --warmup-episodes 3 --train-every 4 --batch-size 128 --replay-size 50000 --gamma 0.95 --lr 1e-4 --lr-min 1e-5 --tau 0.005 --epsilon-start 0.10 --epsilon-end 0.01 --eval-every 3

best one: 
python train_dqn.py --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DQNLargeScale1\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir artifacts/dqn_model --episodes 300 --warmup-episodes 2 --train-every 8 --batch-size 64 --replay-size 100000 --gamma 0.90 --lr 5e-5 --lr-min 1e-6 --tau 0.002 --epsilon-start 0.05 --epsilon-end 0.01 --eval-every 5 --wait-target 254 --ride-time-target 365

python train_dqn.py --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DQNLargeScale1\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir artifacts/dqn_model --episodes 80 --warmup-episodes 1 --train-every 4 --batch-size 256 --replay-size 100000 --gamma 0.95 --lr 5e-5 --lr-min 1e-6 --tau 0.002 --epsilon-start 0.05 --epsilon-end 0.01 --eval-every 4

python train_dqn.py --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DQNLargeScale1\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DQNLargeScale1\RealDQN\DQNetwork\artifacts\dqn_model --episodes 100 --warmup-episodes 2 --epsilon-start 0.1 --epsilon-end 0.01 --lr 1e-4 --lr-min 1e-5 --tau 0.003 --train-every 4 --gamma 0.85 --eval-every 5

python train_dqn.py \
  --cfg <your_cfg> \
  --imitation-model-dir artifacts/imitation_model \
  --output-dir artifacts/dqn_model \
  --episodes 200 \
  --lr 3e-4 \
  --lr-min 1e-5 \
  --gamma 0.8 \
  --tau 0.005 \
  --epsilon-start 0.4 \
  --epsilon-end 0.05 \
  --reward-clip 5.0 \
  --train-every 4 \
  --warmup-episodes 2
  --resume-from artifacts/dqn_model


To test the DQN model: 
python run_dqn_policy.py --cfg D:\FYP\DeepPool\SunwaySmallMap\osm.sumocfg --model-dir D:\FYP\DeepPool\RealDQN\DQNetwork\artifacts\dqn_model


<!-- To analyze the training -->
python analyze_training.py --csv D:\FYP\DQNVersion1\DQNetwork\artifacts\dqn_model\training_history.csv --out results.png
# or with a wider smoothing window:
python analyze_training.py --csv D:\FYP\DQNVersion1\DQNetwork\artifacts\dqn_model\training_history.csv --smooth-window 20


<!--  best DQN training result now -->
{
  "total_requests": 200,
  "completed_requests": 200,
  "completion_rate": 1.0,
  "picked_up_requests": 200,
  "avg_wait_until_pickup": 189.05,
  "max_wait_until_pickup": 550.0,
  "avg_excess_ride_time": 105.306016660960s45,
  "decisions_seen": 200
}

{
  "total_requests": 200,
  "completed_requests": 200,
  "completion_rate": 1.0,
  "picked_up_requests": 200,
  "avg_wait_until_pickup": 205.65,
  "max_wait_until_pickup": 920.0,
  "avg_excess_ride_time": 102.3169778933576,s
  "decisions_seen": 200
}

{
  "total_requests": 200,
  "completed_requests": 200,
  "completion_rate": 1.0,
  "picked_up_requests": 200,
  "avg_wait_until_pickup": 170.25,
  "max_wait_until_pickup": 460.0,
  "avg_excess_ride_time": 106.20685740278765,
  "decisions_seen": 200
}





want the code to focus on create better waiting time and detour time and give other param less weight

7. Only 1 request dispatched per tick
In dqn_env.py:227-231, only the single longest-waiting request gets a decision point per tick. With 2000 requests and 15 taxis, the agent is making sequential decisions that don't account for batch-level optimization. This is fine architecturally, but it means the agent makes ~2000 sequential greedy decisions, and the log shows taxis accumulating 7-11 stops with excess ride times of 1000-1500 seconds — passengers riding 5-6x their direct route time.

python train_dqn.py  --cfg D:\FYP\DQNLargeScale1\SunwaySmallMap\osm.sumocfg --imitation-model-dir artifacts\imitation_model --output-dir artifacts\dqn_model --resume-from artifacts\dqn_model --episodes 150


How to run
Without spatial features (same as before, just faster + dueling + N-step):


python train_dqn.py --cfg path/to/osm.sumocfg --imitation-model-dir ... --output-dir ...
With spatial features (adds 140-dim spatial context):


python train_dqn.py --cfg D:\FYP\DeepPool\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DeepPool\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DeepPool\RealDQN\DQNetwork\artifacts\dqn_model --episodes 150 --warmup-episodes 3 --epsilon-start 0.1 --epsilon-end 0.01 --lr 1e-4 --lr-min 1e-5 --tau 0.003 --train-every 4 --gamma 0.85 --eval-every 5 --n-step 5 --wait-target 254 --ride-time-target 365

python train_dqn.py --cfg D:\FYP\DeepPool\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DeepPool\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DeepPool\RealDQN\DQNetwork\artifacts\dqn_model --episodes 50 --warmup-episodes 3 --epsilon-start 0.15 --epsilon-end 0.02 --lr 1e-4 --lr-min 1e-5 --tau 0.005 --train-every 4 --gamma 0.85 --eval-every 5 --n-step 5 --resume-from D:\FYP\DeepPool\RealDQN\DQNetwork\artifacts\dqn_model --batch-size 128


Note on warm start with spatial features: the first Linear layer goes from 48→256 to 188→256, so it won't load from the imitation model. All deeper layers still warm-start. You may want to add 1-2 extra warmup episodes (--warmup-episodes 3) to compensate.

python train_dqn.py --cfg D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DeepPoolPer_Comparison\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DeepPoolPer_Comparison\RealDQN\DQNetwork\artifacts\dqn_model_015 --episodes 400 --train-every 4 --batch-size 128 --replay-size 40000 --gamma 0.85 --lr 5e-5 --lr-min 1e-6 --tau 0.005 --epsilon-start 0.15 --epsilon-end 0.01 --eval-every 10 --wait-target 244 --ride-time-target 335 --n-step 5 --net D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.net.xml

Exponential Decay: 
python train_dqn.py --cfg D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DeepPoolPer_Comparison\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DeepPoolPer_Comparison\RealDQN\DQNetwork\artifacts\dqn_model_expo --episodes 400 --train-every 4 --batch-size 128 --replay-size 40000 --gamma 0.85 --lr 5e-5 --lr-min 1e-6 --tau 0.005 --epsilon-start 0.325 --epsilon-end 0.01 --eval-every 10 --wait-target 244 --ride-time-target 335 --n-step 5 --net D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.net.xml --epsilon-schedule exponential --epsilon-decay-rate 5 --seed 42

Spatial Encoder: 
python train_dqn.py --cfg D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.sumocfg --imitation-model-dir D:\FYP\DeepPoolPer_Comparison\DQNImitation\ImitationDL\artifacts\imitation_model --output-dir D:\FYP\DeepPoolPer_Comparison\RealDQN\DQNetwork\artifacts\dqn_model_expo --episodes 400 --train-every 4 --batch-size 128 --replay-size 40000 --gamma 0.85 --lr 5e-5 --lr-min 1e-6 --tau 0.005 --epsilon-start 0.325 --epsilon-end 0.01 --eval-every 10 --wait-target 244 --ride-time-target 335 --n-step 5 --net D:\FYP\DeepPoolPer_Comparison\SunwaySmallMap\osm.net.xml --epsilon-schedule exponential --epsilon-decay-rate 5 --seed 42 --spatial-encoder