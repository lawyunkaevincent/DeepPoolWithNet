To train the DQN agent

python train_zone_dqn.py --cfg   ../SunwaySmallMap/osm.sumocfg --net   ../SunwaySmallMap/osm.net.xml --stops ../SunwaySmallMap/stops.json --episodes 50 --epsilon-start 1.0 --epsilon-end 0.1 --epsilon-decay-ep 40

  12am–6am   (very low)  mult=0.10  target=4.9%  generated=195
  6am–9am    (morning peak)  mult=1.00  target=24.4%  generated=976
  9am–4pm    (medium)  mult=0.50  target=28.5%  generated=1138
  4pm–8pm    (evening peak)  mult=1.00  target=32.5%  generated=1301
  8pm–12am   (low to medium)  mult=0.30  target=9.8%  generated=390