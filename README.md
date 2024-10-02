# 状態観測に対する学習された敵対者の交互トレーニングによる強化学習のロバスト性（ATLA）

このリポジトリには、状態観測に対する敵対的攻撃に対する強化学習のロバスト性のための学習された敵対者（ATLA）の交互トレーニングの参照実装が含まれています。

私たちのATLAトレーニング手法は、教師あり学習の「敵対的トレーニング」に多少類似していますが、強化学習エージェントの*最適な*敵対的攻撃を特徴付ける[state-adversarial Markov decision process (SA-MDP)](https://arxiv.org/pdf/2003.08938)に基づいています。トレーニング中に、*最適な*攻撃の定式化に従ってエージェントと一緒に敵対者を学習します。エージェントはトレーニング時にこの強力な敵対者を打ち負かさなければならないため、テスト時には幅広い強力な攻撃に対してロバストになります。以前のアプローチはSA-MDPに基づいておらず、トレーニング中に勾配ベースの攻撃ヒューリスティックを使用していましたが、これらは十分に強力ではなく、テスト時の強力な攻撃に対して脆弱になります。

SA-MDPに従って、エージェントと環境を与えられた場合に**可能な限り低い報酬**を達成する最適な敵対的攻撃を変換されたMDPを解くことで見つけることができます。これは分類問題における**最小の敵対的例**（[MILP/SMTソルバーを使用して見つけることができます](https://arxiv.org/pdf/1709.10207.pdf)）に類似しています。DRL設定では、このMDPはPPOなどの任意のDRLアルゴリズムを使用して解くことができます。最適な敵対者フレームワークに基づく敵対的攻撃が以前に提案された強力な攻撃よりも大幅に強力であることを実証します（[下の例を参照](#optimal-adversarial-attack-and-atla-ppo-demo)）。最適な攻撃フレームワークは、将来開発される防御技術に対するRLエージェントのロバスト性を評価するのに役立ちます。

RLに対する最適な敵対的攻撃とATLAトレーニングフレームワークの詳細は、私たちの論文に記載されています：

"**Robust Reinforcement Learning on State Observations with Learned Optimal Adversary**",  
[Huan Zhang](http://huan-zhang.com) (UCLA), [Hongge Chen](https://scholar.google.com/citations?user=KFtsQvIAAAAJ&hl=en) (MIT), 
[Duane Boning](https://www-mtl.mit.edu/wpmu/researchgroupsboning/boning/) (MIT), [Cho-Jui Hsieh](http://web.cs.ucla.edu/~chohsieh/) (UCLA) (\* Equal contribution)  
ICLR 2021. [(Paper PDF)](https://arxiv.org/pdf/2101.08452.pdf)

私たちのコードはSA-PPOロバスト強化学習コードベースに基づいています：
[huanzhang12/SA_PPO](https://github.com/huanzhang12/SA_PPO)。

## 最適な敵対的攻撃とATLA-PPOデモ

論文ではまず、SA-MDPの*最適な*敵対的攻撃設定の下で敵対者を学習できることを示しています。これにより、RLエージェントを攻撃するための大幅に強力な敵対者を得ることができます：以前の強力な攻撃ではエージェントが動けなくなる一方で、私たちの学習された敵対者はエージェントを**反対方向に動かす**ことができ、**大きなマイナスの報酬**を得ます。さらに、私たちのATLAフレームワークを使用してこの強力な敵対者でトレーニングすることで、エージェントは強力な敵対的攻撃に対してロバストになります。

| | バニラPPO <br> 攻撃なし | バニラPPO <br> ロバストSarsa (RS) <br> 攻撃下 | バニラPPO <br> 学習された最適な攻撃下 | 私たちのATLA-PPO <br> 学習された最適な攻撃下 <br> (最強の攻撃) |
|:--:|:--:| :--:| :--:| :--:|
| Ant-v2 | ![ant_ppo_natural_5358.gif](/assets/ant_ppo_natural_5358.gif) | ![ant_ppo_rs_attack_63.gif](/assets/ant_ppo_rs_attack_63.gif) | ![ant_ppo_optimal_attack_-1141.gif](/assets/ant_ppo_optimal_attack_-1141.gif) | ![ant_atla_ppo_optimal_attack_3835.gif](/assets/ant_atla_ppo_optimal_attack_3835.gif) |
| エピソード <br> 報酬 | **5358** <br> 右へ移動 ➡️ | **63** <br> 動かない 🛑 | **-1141** <br> 左へ移動 ◀️  <br> (目標の反対) | **3835** <br> 右へ移動 ➡️ |
| **HalfCheetah-v2** | ![halfcheetah_ppo_natural_7094.gif](/assets/halfcheetah_ppo_natural_7094.gif) | ![halfcheetah_ppo_rs_attack_85.gif](/assets/halfcheetah_ppo_rs_attack_85.gif) | ![halfcheetah_ppo_optimal_attack_-743.gif](/assets/halfcheetah_ppo_optimal_attack_-743.gif) | ![halfcheetah_ppo_natural_7094.gif](/assets/halfcheetah_atla_ppo_optimal_attack_5250.gif) |
| エピソード <br> 報酬 | **7094** <br> 右へ移動 ➡️ | **85** <br> 動かない 🛑 | **-743** <br> 左へ移動 ◀️  <br> (目標の反対) | **5250** <br> 右へ移動 ➡️ |

## セットアップ

まず、このリポジトリをクローンし、必要なPythonパッケージをインストールします：

```bash
git submodule update --init
pip install -r requirements.txt
sudo apt install parallel  # 最適な攻撃実験を実行するためにのみ必要です。
cd src  # すべてのコードファイルはsrc/フォルダにあります
```

OpenAI Gym環境を使用するために、まずMuJoCo 1.5をインストールする必要があります。
インストール手順については[こちら](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/README.md#requirements)を参照してください。

## 事前学習済みエージェント

私たちは論文で評価されたすべての設定について事前学習済みエージェントを公開しています。これらの事前学習済みエージェントは `src/models/atla_release` にあり、6つの設定に対応する6つのサブディレクトリがあります。各フォルダ内には、エージェントモデル（`model-`から始まる）と最適な敵対的攻撃のために学習された敵対者（`attack-`から始まる）が含まれています。後のセクションでこれらのモデルをロードする方法を示します。事前学習済みエージェントのパフォーマンスは以下のとおりです。ここでは最強のATLA-PPO（LSTM + SA-Reg）メソッドと強力なベースラインSA-PPO、およびロバストトレーニングを行わないバニラPPOの自然エピソード報酬と提案する最適な攻撃下でのエピソード報酬を報告します。完全な結果とより多くのベースラインについては[論文](https://arxiv.org/pdf/2101.08452.pdf)を参照してください。

| 環境            | 評価              | バニラPPO | SA-PPO         | ATLA-PPO (LSTM + SA-Reg) |
|------------------|-----------------|-------------|----------------|--------------------------|
| Ant-v2           | 攻撃なし          |  5687.0    |      4292.1    |       5358.7             |
|                  | 最強の攻撃         |   -871.7     |    2511.0      |         **3764.5**           |
| HalfCheetah-v2   | 攻撃なし          |      7116.7       |       3631.5         |             6156.5             |
|                  | 最強の攻撃         |      -660.5       |      3027.9          |              **5058.2**             |
| Hopper-v2      | 攻撃なし          |        3167.3     |        3704.5        |             3291.2             |
|                  | 最強の攻撃         |        636.4     |        1076.3        |            **1771.9**              |
| Walker2d-v2        | 攻撃なし          |      4471.7       |        4486.6        |               3841.7           |
|                  | 最強の攻撃         |    1085.5  |         2907.7       |              **3662.9**            |

強化学習アルゴリズムは通常、トレーニング実行間で大きな分散を持つことに注意してください。そのため、各エージェント構成を21回繰り返しトレーニングし、6つの攻撃の中で最強の（最良の）攻撃下での50エピソードの平均累積報酬でランク付けします。事前学習済みエージェントは**中央値のロバスト性**（最強の攻撃下でのエピソード報酬の中央値）を持つものであり、最良のものではありません。私たちの作業と比較する際には、**各エージェントを少なくとも10回繰り返しトレーニングし、中央値のエージェントを報告することが重要です**。さらに、ロバストサルサ（RS）攻撃と提案する最適な攻撃に対して、多くの攻撃パラメータを検索し、その中で最強の敵対者を選びます。詳細については[下記セクション](#optimal-attack-to-deep-reinforcement-learning)を参照してください。

事前学習済みエージェントは `test.py` を使用して評価できます（使用方法の詳細については次のセクションを参照）。例：

```bash
# Antエージェント。
## バニラPPO：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic
## SA-PPO：
python test.py --config-path config_ant_sappo_convex.json --load-model models/atla_release/SAPPO/model-sappo-convex-ant.model --deterministic
## バニラLSTM：
python test.py --config-path config_ant_vanilla_ppo_lstm.json --load-model models/atla_release/LSTM-PPO/model-lstm-ppo-ant.model --deterministic
## ATLA PPO（MLP）：
python test.py --config-path config_ant_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-ant.model --deterministic
## ATLA PPO（LSTM）：
python test.py --config-path config_ant_atla_ppo_lstm.json --load-model models/atla_release/ATLA-LSTM-PPO/model-lstm-atla-ppo-ant.model --deterministic
## ATLA PPO（LSTM+SA Reg）：
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --deterministic
```

**--deterministic**スイッチは重要であり、評価のために確率的行動を無効にします。`ant` を `walker`、`hopper` または `halfcheetah` に変更して、他の環境を試すこともできます。

## 深層強化学習への最適な攻撃
### 単一の最適攻撃敵対者のトレーニング
最適攻撃を実行するには、`--mode` を `adv_ppo` に設定し、`--ppo-lr-adam` をゼロに設定します。これは、エージェントモデルの学習率を0に設定して私たちのATLAトレーニングを実行するのと本質的に同じです。このようにして敵対者のみを学習します。敵対者のポリシーネットワークの学習率は `--adv-ppo-lr-adam` で設定でき、価値ネットワークの学習率は `--adv-val-lr` で設定でき、敵対者のエントロピーレギュラライザーは `--adv-entropy-coeff` で設定でき、敵対者のPPOオプティマイザーのクリッピングイプシロンは `--adv-clip-eps` で設定できます。

```bash
# 注：これは説明のみを目的としています。敵対者のハイパーパラメータを通常はハイパーパラメータサーチによって正しく選択する必要があります。
python run.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --mode adv_ppo --ppo-lr-adam 0.0 --adv-ppo-lr-adam 3e-5 --adv-val-lr 3e-5 --adv-entropy-coeff 0.0 --adv-clip-eps 0.4
```
これにより、`vanilla_ppo_ant/agents/YOUR_EXP_ID` に実験フォルダが保存されます。ここで、`YOUR_EXP_ID` はランダムに生成された実験IDです（例：`e908a9f3-0616-4385-a256-4cdea5640725`）。このフォルダから最良のモデルを抽出するには、

```bash
python get_best_pickle.py vanilla_ppo_ant/agents/YOUR_EXP_ID
```
を実行し、`best_model.YOUR_EXP_ID.model` という名前の敵対者モデル（例：`best_model.e908a9f3.model`）を生成します。

次に、トレーニングされたこの敵対者を評価するには、

```bash
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network best_model.YOUR_EXP_ID.model
```
を実行します。

### 最適な敵対的攻撃敵対者の発見
上記のコマンドは、1つの敵対者を1つの敵対者ハイパーパラメータセットを使用してトレーニングおよびテストするだけです。この最適な敵対者の学習も強化学習問題（PPOを使用して解決）であるため、攻撃結果を最適化しエージェントモデルの真のロバスト性を評価するためには、複数の敵対者ハイパーパラメータセットを使用して敵対者をトレーニングし、最強の（最良の）敵対者を選ぶ必要があります。私たちは、敵対者のハイパーパラメータを簡単にスキャンし、各ハイパーパラメータセットを並行して実行するためのスクリプトを提供しています：

```bash
cd ../configs
# これにより、agent_configs_attack_ppo_antフォルダ内に216の構成ファイルが生成されます。
# 攻撃_ppo_ant_scan.pyを変更して、グリッドサーチのハイパーパラメータを変更します。
# 通常、異なる環境では異なるハイパーパラメータセットが必要です。
python attack_ppo_ant_scan.py
cd ../src
# このコマンドは利用可能なすべてのCPUを使用して216の構成を実行します。

# すべてのCPUを使用したくない場合は、「-t」を使用してスレッド数を制御できます。
python run_agents.py ../configs/agent_configs_attack_ppo_ant_scan/ --out-dir-prefix=../configs/agents_attack_ppo_ant_scan > attack_ant_scan.log
```

上記のトレーニングコマンドが完了したら、学習された最適な攻撃敵対者の評価スクリプトを実行するだけです：

```bash
bash example_evaluate_optimal_attack.sh
```

最適な攻撃敵対者の評価を別の環境や別のフォルダの結果に対して実行する場合は、`example_evaluate_optimal_attack.sh`内の `scan_exp_folder`行を変更する必要があることに注意してください。この行を次のように変更します：

```bash
scan_exp_folder <config file> <path to trained optimal attack adversarial> <path to the victim agent model> $semaphorename
```

このスクリプトは、評価を並行して実行し（"GNU parallel"ツールが必要）、各実験IDフォルダに攻撃結果を含むログファイル `attack_scan/optatk_deterministic.log` を生成します。上記のコマンドが完了した後、`parse_optimal_attack_results.py` を使用してログを解析し、最強（最適）の攻撃結果（エージェント報酬が最低）を取得できます：

```bash
python parse_optimal_attack_results.py ../configs/agents_attack_ppo_ant_scan/attack_ppo_ant/agents
```

最適な敵対的攻撃を実施する場合、攻撃自体もRL問題であり、ハイパーパラメータに敏感であるため、上記のようなハイパーパラメータサーチスキームを使用することが重要です。エージェントの真のロバスト性を評価するためには、最適な攻撃敵対者を見つけることが必要です。

### すべてのエージェントの事前学習済み敵対者

私たちは、公開したすべてのエージェントの最適な攻撃敵対者を提供します。事前学習済みの最適攻撃敵対者をテストするには、`test.py` を `--attack-advpolicy-network` オプションとともに実行します：

```bash
# Antエージェント。
## バニラPPO：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/PPO/attack-ppo-ant.model
## SA-PPO：
python test.py --config-path config_ant_sappo_convex.json --load-model models/atla_release/SAPPO/model-sappo-convex-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/SAPPO/attack-sappo-convex-ant.model
## バニラLSTM：
python test.py --config-path config_ant_vanilla_ppo_lstm.json --load-model models/atla_release/LSTM-PPO/model-lstm-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/LSTM-PPO/attack-lstm-ppo-ant.model
## ATLA PPO（MLP）：
python test.py --config-path config_ant_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-ant.model
## ATLA PPO（LSTM）：
python test.py --config-path config_ant_atla_ppo_lstm.json --load-model models/atla_release/ATLA-LSTM-PPO/model-lstm-atla-ppo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-LSTM-PPO/attack-lstm-atla-ppo-ant.model
## ATLA PPO（LSTM+SA Reg）：
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --deterministic --attack-method advpolicy --attack-advpolicy-network models/atla_release/ATLA-LSTM-SAPPO/attack-atla-lstm-sappo-ant.model
```

configファイル名、エージェントモデルファイル名、敵対者モデルファイル名で `ant` を `walker`、`hopper` または `halfcheetah` に変更して、他の環境を試すことができます。

## 学習された最適な敵対者によるエージェントのトレーニング（ATLAフレームワーク）

エージェントをトレーニングするには、`src`フォルダの `run.py` を使用し、構成ファイルパスを指定します。いくつかの構成ファイルは `src` フォルダに提供されており、ファイル名は `config` で始まります。例えば：

HalfcheetahバニラPPO（MLP）トレーニング：

```bash
python run.py --config-path config_halfcheetah_vanilla_ppo.json
```

HalfCheetahバニラPPO（LSTM）トレーニング：

```bash
python run.py --config-path config_halfcheetah_vanilla_ppo_lstm.json
```

HalfCheetah ATLA（MLP）トレーニング：

```bash
python run.py --config-path config_halfcheetah_atla_ppo.json
```

HalfCheetah ATLA（LSTM）トレーニング：

```bash
python run.py --config-path config_halfcheetah_atla_ppo_lstm.json
```

状態敵対的正則化器を使用したHalfCheetah ATLA（LSTM）トレーニング（これは最良の方法です）：

```bash
python run.py --config-path config_halfcheetah_atla_lstm_sappo.json
```

`halfcheetah` を `ant`、`hopper` または `walker` に変更して他の環境を実行します。

トレーニング結果は、jsonファイルの `out_dir` パラメータで指定されたディレクトリに保存されます。例えば、状態敵対的正則化器を使用したATLA（LSTM）トレーニングの場合、`robust_atla_ppo_lstm_halfcheetah` です。複数の実行を許可するために、各実験には一意の実験ID（例：`2fd2da2c-fce2-4667-abd5-274b5579043a`）が割り当てられ、`out_dir`（例：`robust_atla_ppo_lstm_halfcheetah/agents/2fd2da2c-fce2-4667-abd5-274b5579043a`）に保存されます。

次に、エージェントは `test.py` を使用して評価できます。例えば：

```bash
# robust_atla_ppo_lstm_halfcheetah/agents/ のフォルダ名に一致する --exp-id を変更します。
python test.py --config-path config_halfcheetah_atla_lstm_sappo.json --exp-id YOUR_EXP_ID --deterministic
```

ほとんどのメソッドで50エピソードの平均累積報酬が5000を超えることを期待すべきです。

## 攻撃下でのエージェントの評価

私たちはランダム攻撃、クリティックベースの攻撃、および提案するロバストサルサ（RS）および最大アクション差（MAD）攻撃を実装しました。

### 最適な敵対的攻撃

最適な敵対的攻撃を実行する方法の詳細については、[このセクション](#optimal-attack-to-deep-reinforcement-learning)を参照してください。これは現在最強の攻撃であり、RL防御アルゴリズムのロバスト性を評価するために強く推奨されます。

### ロバストサルサ（RS）攻撃

私たちのロバストサルサ攻撃では、まず評価対象のポリシーに対して*ロバスト*な価値関数を学習します。次に、このロバストな価値関数を使用してポリシーを攻撃します。RS攻撃の最初のステップはロバストな価値関数のトレーニングです（例としてAnt環境を使用）：

```bash
# ステップ1：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --sarsa-enable --sarsa-model-path sarsa_ant_vanilla.model
```

上記のトレーニングステップは通常非常に高速です（数分程度）。価値関数は `sarsa_ant_vanilla.model` に保存されます。次にこれを攻撃に使用します：

```bash
# ステップ2：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method sarsa --attack-sarsa-network sarsa_ant_vanilla.model --deterministic
```

攻撃のL無限大ノルムは `--attack-eps` パラメータで設定されます（異なる環境では異なる攻撃のイプシロンが必要です。詳細は論文のTable 2を参照）。報告される50エピソードの平均報酬は500未満であるべきです（攻撃なしの報酬は5000以上です）。対照的に、私たちのATLA-PPO（LSTM + SA-Reg）ロバストエージェントはこの特定の攻撃下でも報酬が4000以上あります：

```bash
# ロバストな価値関数をトレーニング。
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --sarsa-enable --sarsa-model-path sarsa_ant_atla_lstm_sappo.model
# ロバストな価値関数を使用して攻撃。
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --attack-eps=0.15 --attack-method sarsa --attack-sarsa-network sarsa_ant_atla_lstm_sappo.model --deterministic
```

ロバストサルサ攻撃には、ロバスト性正則化のための2つのハイパーパラメータ（`--sarsa-eps` および `--sarsa-reg`）があり、ロバストな価値関数を構築します。デフォルト設定は通常良好に機能しますが、包括的なロバスト性評価のためには、異なるハイパーパラメータの下でロバストサルサ攻撃を実行し、最良の攻撃（最低の報酬）を最終結果として選択することが推奨されます。包括的な敵対的評価のために `scan_attacks.sh` スクリプトを提供しています：

```bash
# まずGNU parallelをインストールする必要があります：sudo apt install parallel
source scan_attacks.sh
# 使用法：scan_attacks model_path config_path output_dir_path
scan_attacks models/atla_release/PPO/model-ppo-ant.model config_ant_vanilla_ppo.json sarsa_ant_vanilla_ppo_result
```

上記の例では、スクリプトによって報告される「最小のRS攻撃報酬（決定的なアクション）」が300未満であることを確認するべきです。ご参考までに、`scan_attacks.sh` スクリプトは、MAD攻撃、クリティック攻撃、およびランダム攻撃を含む他の多くの攻撃も実行します。ロバストサルサ攻撃は通常その中で最も強力なものです。

注：サルサモデルの学習率は `--val-lr` で変更できます。デフォルト値は提供されている環境（正規化された報酬）に対して良好に機能するはずです。しかし、この攻撃を別の環境で使用したい場合、この学習率は重要です（いくつかの環境は大きな報酬を返し、Q値が大きくなり、大きな `--val-lr` が必要です）。経験則として、これらのサルサモデルのトレーニングログを常に確認することが重要です。トレーニング終了時にQ損失が十分に減少していることを確認してください（0に近い）。

### 最大アクション差（MAD）攻撃

さらに、最大アクション差（MAD）攻撃を提案し、元のアクションと摂動されたアクションの間のKLダイバージェンスを最大化しようとします。これを実行するには、`--attack-method` を `action` に設定します。例えば：

```bash
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method action --deterministic
```

報告される50エピソードの平均報酬は約1500であるべきです（この攻撃はこの場合、ロバストサルサ攻撃よりも弱いです）。対照的に、私たちのATLA-PPO（LSTM + SA-Reg）ロバストエージェントはMAD攻撃に対してより耐性があり、報酬が5000以上になります。

```bash
python test.py --config-path config_ant_atla_lstm_sappo.json --load-model models/atla_release/ATLA-LSTM-SAPPO/model-atla-lstm-sappo-ant.model --attack-eps=0.15 --attack-method action --deterministic
```

さらに、RS+MADの組み合わせ攻撃を提供しており、`--attack-method` を `sarsa+action` に設定し、組み合わせ比率を `--attack-sarsa-action-ratio` で0から1の範囲で設定できます。

### クリティックベース攻撃とランダム攻撃

クリティックベース攻撃とランダム攻撃は、それぞれ `--attack-method` を `critic` および `random` に設定することで使用できます。これらの攻撃は比較的弱く、PPOエージェントのロバスト性を評価するのには適していません。

```bash
# クリティックベース攻撃（Pattanaik et al.）
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method critic --deterministic
# ランダム攻撃（均一ノイズ）
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method random --deterministic
```

この場合、クリティックまたはランダム攻撃下ではエージェント報酬は5000以上のままであり、これらの攻撃がこの特定の環境では非常に効果的ではないことを意味します。

### スヌーピング攻撃

このリポジトリでは、[Inkawhich et al.](https://arxiv.org/pdf/1905.11832.pdf)によって提案された模倣学習ベースのスヌーピング攻撃も実装しています。この攻撃では、まず評価対象のポリシーから新しいエージェントを学習します。次に、この新しいエージェントの勾配情報を使用して元のポリシーを攻撃します。スヌーピング攻撃の最初のステップは、元のエージェントの行動を観察（「スヌーピング」）して新しい模倣エージェントをトレーニングすることです：

```bash
# ステップ1：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --imit-enable --imit-model-path imit_ant_vanilla.model
```

上記のトレーニングステップは通常非常に高速です（数分程度）。新しいエージェントモデルは `imit_ant_vanilla.model` に保存されます。次にこれを読み込んでスヌーピング攻撃を実行します：

```bash
# ステップ2：
python test.py --config-path config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-eps=0.15 --attack-method action+imit --imit-model-path imit_ant_vanilla.model --deterministic
```

スヌーピング攻撃はブラックボックス攻撃（エージェントポリシーの勾配やエージェントとの相互作用を必要としない）であるため、他のホワイトボックス攻撃よりも通常は弱いです。上記の例では、平均エピソード報酬はおおよそ3000であるべきです。