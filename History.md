# 問題1 テスト時に訓練時の時の性能が出ない
原因 : 環境を作成するたびにseed値らしいものが変わってしまっている
解決 : 環境のリセット時に環境を新しく作成した

# 問題2 SA-RLforIAの性能が出ない
解決 : 被害者モデルを決定論的にすることで解決した

# 問題3 TD3アルゴリズムのQ lossが低すぎる。正しく学習されるならもっと大きい値になるはず
解決 : Envクラスでreward normalizationが行われていた。 victimの環境を用いているため、reward normalizationがvictim学習時のものを用いており、初期のrewardが非常に小さくなってしまっていた。MetaWorldは学習が進みやすいように調整されているため、normalizedされていない報酬を用いるようにした。
しかし、性能は上がらず。Q lossが小さいことは性能とは無関係？
けど、前回の実装ではQ lossは300程度まで上がっていたのに対して、今回は2,30程度しか上がっていない。この原因は？

# 問題4 RAIDENでpip installをするとエラーが出る
解決 : proxyを設定することで解決。RAIDENドキュメントに記載されていた以下のproxyを環境変数に設定することで、proxy経由でのpip installに対応
export http_proxy=http://10.1.10.1:8080/
export https_proxy=http://10.1.10.1:8080/

# 問題5 RAIDENでgtn-containerにログインしたときに環境変数などが設定されない
解決 : ~/.bash_profileに記載した内容が間違っていた。if文の条件式は dl-g* にする必要がある(gpuやgtnなどの場合があるため)
また、nvidia_entrypoint.shの場所も変える必要あり
/usr/local/bin/nvidia_entrypoint.sh -> /opt/nvidia/nvidia_entrypoint.sh
```
if [[ $HOSTNAME == dl-gpu* ]] ; then
 CONTAINER=`cat ${HOME}/now_container.dat`
 echo $CONTAINER
 /usr/local/bin/nvidia_entrypoint.sh &
 pid=$!
 wait $pid
 . /fefs/opt/dgx/env_set/$CONTAINER.sh

 export MY_PROXY_URL="http://10.1.10.1:8080/"
 export HTTP_PROXY=$MY_PROXY_URL
 export HTTPS_PROXY=$MY_PROXY_URL
 export FTP_PROXY=$MY_PROXY_URL
 export http_proxy=$MY_PROXY_URL
 export https_proxy=$MY_PROXY_URL
 export ftp_proxy=$MY_PROXY_URL

 mkdir -p ~/.raiden/$CONTAINER
 export PATH="${HOME}/.raiden/$CONTAINER/bin:$PATH"
 export LD_LIBRARY_PATH="${HOME}/.raiden//$CONTAINER/lib:$LD_LIBRARY_PATH"
 export LDFLAGS=-L/usr/local/nvidia/lib64
 export PYTHONPATH="${HOME}/.raiden/$CONTAINER/lib/python3.9/site-packages"
 export PYTHONUSERBASE="${HOME}/.raiden/$CONTAINER"
 export PREFIX="${HOME}/.raiden/$CONTAINER"
fi
```

# 問題6 git cloneできない
解決 : Personal access tokensにより解決
token: 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

# 問題7 ILfDにおいてscoreが0になったり3000になったりと不安定
一部解決 : demoが更新されていなかったので、新しいエキスパートのやつにしたら少し改善した

# 2024/08/18 window-close-v2に関するロバストなモデルを作成
discounted_robust_ppo : 004を使用. robust_ppo_reg = 1.0
sappo_convex : 002を使用. robust_ppo_reg = 0.5
atla_ppo : 005を使用. adv_entropy_coeff=3e-4, adv_ppo_lr_adam=1e-3

# 2024/08/18 window-open-v2に関するロバストなモデルを作成
discounted_robust_ppo : 004を使用. robust_ppo_reg = 1.0
sappo_convex : 002を使用. robust_ppo_reg = 0.5
atla_ppo : 005を使用. adv_entropy_coeff=3e-4, adv_ppo_lr_adam=1e-3

# 2024/08/18 window-close-v2モデルに対するILfD, ILfOモデルを作成
ILfD : 030を使用. adv_ppo_lr_adam = 2e-4, gradient_steps = 250, disc_gradient_steps = 1
ILfO : 020を使用. adv_ppo_lr_adam = 2e-4, gradient_steps = 1000, disc_gradient_steps = 1

# 問題8 SA-PPOとDiscounted_robust_ppoのどちらもSAIAに対して防御性能が高すぎて、差がつかない

# 2024/08/19 Vanilla PPOの性能をtest.pyで評価
vanilla_ppo, window-close-v2, natural: 
mean: 4433.784546657522, std:108.40628156111546, min:4143.580884778766, max:4579.322247755346
vanilla_ppo, attack_obj : window-close-v2, random:
mean: 482.83835087027654, std:3.744291606590614, min:475.0383231221688, max:488.54586290882384 
vanilla_ppo, attack_obj : window-close-v2, ppo_ia(011):
mean: 3920.2678057752964, std:1219.2462306590166, min:481.47199494078455, max:4538.226575143001
vanilla_ppo, attack_obj : window-close-v2, ilfd(017):
mean: 3963.6800658289158, std:519.6748926214161, min:1649.4982536333428, max:4503.300304116634
vanilla_ppo, attack_obj : window-close-v2, ilfo(006):
mean: 3908.5984161335928, std:480.9537022236074, min:2841.9041500015837, max:4581.366714467547

vanilla_ppo, window-open-v2, natural:
mean: 3603.298627841156, std:1115.1198015675664, min:274.9850639470146, max:4439.446591741157
(mean: 4403, std 115)
vanilla_ppo, attack_obj : window-open-v2, random:
mean: 315.8083864055749, std:95.25699824169877, min:165.0950505740855, max:777.5693278662882
vanilla_ppo, attack_obj : window-open-v2, ppo_ia(010):
mean: 766.5826018518221, std:722.5175402624664, min:171.77046968355378, max:2616.9194860615717
vanilla_ppo, attack_obj : window-open-v2, ilfd(017):
mean: 917 std: 395
vanilla_ppo, attack_obj : window-open-v2, ilfo(017):
mean: 830 std: 293

# 2024/08/19 Atla_lstm PPOの性能をtest.pyで評価
atla_lstm_ppo, window-close-v2, natural:
mean: 4512.194472012836, std:70.26961610851927, min:4210.9967661040255, max:4604.312362643654
atla_ppo_lstm, attack_obk : window-close-v2, random:
mean: 802.7427338550165, std:738.911441269018, min:472.8661285538319, max:3498.90439363134
atla_lstm_ppo, attack_obj : window-close-v2, ppo_ia(003):
mean: 1703.8080650417635, std:1564.547985813417, min:435.4675062376639, max:4226.542881858777
atla_lstm_ppo, attack_obj : window-close-v2, ilfd(010):
mean: 2694.0048751312283, std:1259.9246327605863, min:305.54961046430105, max:4025.9856257266583
atla_lstm_ppo, attack_obj : window-close-v2, ilfo(008):
mean: 2753.385273312555, std:1342.3442672743581, min:454.5123093754108, max:4007.075797240922


atla_lstm_ppo, window-open-v2, natural:
mean: 3282.4940860993793, std:1600.9070747637365, min:574.9384654355739, max:4488.462616235942
atla_ppo_lstm, attack_obj : window-open-v2, random:
mean: 172.51940807027174, std:40.35469583329893, min:79.63055432403671, max:239.99423031712982
atla_lstm_ppo, attack_obj : window-open-v2, ppo_ia(011):
mean: 561.600867933525, std:537.7773222009909, min:158.77863736355224, max:2191.123348983728
atla_lstm_ppo, attack_obj : window-close-v2, ilfd(010):
mean: 721 std: 352
atla_lstm_ppo, attack_obj : window-close-v2, ilfo(008):
mean: 620 std 428

# 2024/08/20 robust_ppo の性能をtest.pyで評価
robust_ppo(006), window-close-v2, natural:
mean: 4014.984797938935, std:200.32005108076467, min:3614.3105870582526, max:4291.424113893754
robust_ppo, attack_obj : window-close-v2, random:
mean: 478.44652021708464, std:4.007138484564818, min:468.1737905876344, max:484.98146511780425
robust_ppo, attack_obj : window-close-v2, ppo_ia(011):
mean: 470.53878603627527, std:17.84766588582663, min:414.4247028925059, max:484.69581220872084
robust_ppo, attack_obj : window-close-v2, ilfd(017):
mean: 504.4658937173379, std:74.053189869603, min:338.34370915623583, max:666.7506934110811
robust_ppo, attack_obj : window-close-v2, ilfo(010):
mean: 477.6247664508313, std:63.424943848221254, min:380.69079853401513, max:731.1118589413602


robust_ppo(006), window-open-v2, natural:
mean: 4301.506311645386, std:101.38584381260178, min:4025.51123671905, max:4439.386783609647
(3992, 301)
robust_ppo, attack_obj : window-open-v2, random:
mean: 205.18428905691343, std:50.57010022698592, min:85.45797500321474, max:294.26169875790123
robust_ppo, attack_obj : window-open-v2, ppo_ia(011):
mean: 212.0379040980049, std:50.161989166799565, min:88.22107087176241, max:300.76655791939504
robust_ppo, attack_obj : window-open-v2, ilfd(017):
mean: 145.9291543702516, std:39.28606627227373, min:62.80455024351772, max:257.87970249431277
robust_ppo, attack_obj : window-open-v2, ilfo(017):
mean: 144.77492804079577, std:38.40851351736484, min:62.80947128420583, max:257.5578487719442


# 2024/08/20 discounted_robust_ppo の性能をtest.pyで評価
discounted_robust_ppo(006), window-close-v2, natural:
mean: 4397.890286132232, std:57.720788034708676, min:4261.703551138274, max:4498.2946501409815
discounted_robust_ppo, attack_obj : window-close-v2, random:
mean: 477.7909753647112, std:4.799337353847042, min:466.9395839806771, max:485.48129188002855
discounted_robust_ppo, attack_obj : window-close-v2, ppo_ia(011):
mean: 480.0003596181392, std:3.7940032824238448, min:471.6205536649603, max:485.9582907522964
discounted_robust_ppo, attack_obj : window-close-v2, ilfd(008):
mean: 414.4418549341953, std:79.2431662335104, min:219.77876310634045, max:577.2059731973956
discounted_robust_ppo, attack_obj : window-close-v2, ilfo(008):
mean: 482.6700674358357, std:3.4181581815153548, min:474.8904086054135, max:487.85206309214215

discounted_robust_ppo(006), window-open-v2, natural:
mean: 4101.859222757379, std:456.24662481247503, min:2457.1072622432807, max:4466.801253625255
(4301, 256)
discounted_robust_ppo, attack_obj : window-open-v2, random:
mean: 227.1122696490082, std:50.69809777125045, min:97.75907554296376, max:303.99927395879735
discounted_robust_ppo, attack_obj : window-open-v2, ppo_ia(011):
mean: 224.30370540840772, std:51.50617049950561, min:99.06271407049027, max:301.1781506018446
discounted_robust_ppo, attack_obj : window-open-v2, ilfd(008):
mean: 176.55219973030202, std:53.28417961683098, min:62.803626796127375, max:276.24857962261825
discounted_robust_ppo, attack_obj : window-open-v2, ilfo(008):
mean: 169.2637870851453, std:44.281238829177234, min:68.04052640924353, max:258.0090697612437


# 2024/08/25 vanilla_ppo evaluation
window-close(029)
mean: 4542.51788973464, std:39.024755665495384, min:4423.311498450293, max:4615.048515772398
success rate 1.0
window-open(009)
mean: 4508.140198614, std:120.79258888540573, min:4197.379319215332, max:4676.597779409949
success rate 0.98
drawer-close(005)
mean: 4868.00876454665, std:6.2719740513855635, min:4830.438227332478, max:4870.0
success rate 1.0
drawer-open(013)
mean: 4713.853202268044, std:15.856486345360546, min:4684.327024359457, max:4738.502552728423
success rate 1.0

# 2024/09/19 discounted_robust_ppo evaluation
window-close(014)

window-open(014)

drawer-close(014)

drawer-open(014)

# 2024/09/19 sappo_convex evaluation
window-close(014)

window-open(014)

drawer-close(014)

drawer-open(014)



# How to use collect_demo.py
```
python collect_demo.py --config-path config_window-close-v2_vanilla_ppo.json --load-model models/vanilla_ppo/vanilla-ppo-window-close-v2.model --deterministic
```

# How to evaluate attack performance
```
python test.py --config-path {adversary_config}.json --load-model {victim_model}.model --deterministic --attack-method advpolicy --attack-advpolicy-network {adversary_model}.model
```