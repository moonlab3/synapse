# Synapse

Synapse는 ROS 2 기반의 로봇 동작 실행 프레임워크입니다. 관측(카메라 이미지와 관절 상태)을 Brain 어댑터에 전달해 액션을 얻고, 선택한 Muscle 백엔드(시뮬레이터·더미·실기기)에 명령을 내립니다. Behavior Tree 시나리오 실행, 수동 조작, UI(TUI/Qt GUI), 다중 Brain 선택도 지원합니다.

> 이 저장소는 ROS 2 패키지 `synapse`입니다. 현재 설정과 코드에서 확인한 기능을 기준으로 작성했습니다. 실제 로봇이나 Isaac Sim 연결에는 해당 장비/시뮬레이터 드라이버와 토픽 설정이 별도로 필요합니다.

## 무엇에 사용할 수 있나요?

- **로봇 정책 연결**: 원격 GR00T 정책 서버에 관측을 보내고 액션 청크를 받아 실행합니다.
- **사람의 원격 조작**: MANUAL 어댑터로 UI에서 조작하거나 AVP(Tracking Streamer) 추적 데이터를 로봇 좌표에 매핑해 조작합니다.
- **동작 시나리오 실행**: YAML 시나리오를 Behavior Tree로 실행하고 MOVE 및 기록 동작을 조합합니다.
- **재생·시험**: ROS bag JSON 기록을 재생하거나 RANDOM/DUMMY Brain 및 DUMMY Muscle로 파이프라인을 시험합니다.
- **다른 하드웨어/정책 확장**: 공통 Brain 어댑터 인터페이스에 맞춰 어댑터를 추가할 수 있습니다.

구성 요소의 역할은 다음과 같습니다.

| 구성 요소 | 역할 |
| --- | --- |
| Brain | 관측을 받아 액션을 생성하는 정책/입력 어댑터 |
| Synapse 메인 노드 | ROS 2 관측 수집, Brain 선택, 추론, 액션 발행 및 시나리오 실행 |
| Muscle | 액션을 시뮬레이터, 더미 노드 또는 실제 로봇으로 전달 |
| Embodiment 설정 | 로봇 관절, 상태/명령 토픽, 카메라 등 로봇 모델 정의 |

## 요구 사항 및 빌드

저장소에는 ROS 2 패키지 메타데이터와 Python 의존성이 포함되어 있습니다. ROS 2 배포판, Isaac Sim, 로봇 드라이버는 환경에 맞게 설치해야 합니다. Python 의존성 목록은 `src/synapse/setup.py`, ROS 의존성은 `src/synapse/package.xml`을 참고하세요.

ROS 2 워크스페이스에서 패키지를 빌드하는 기본 흐름:

```bash
# 워크스페이스 루트에서 저장소를 src 아래에 둔 뒤
colcon build --symlink-install --packages-select synapse
source install/setup.bash
```

이 저장소의 `scripts` 파일에는 특정 ROS 컨테이너와 Allegro V6 패키지를 위한 예시 명령이 있으나, 일반 설치 스크립트는 아닙니다.

## 실행

패키지 설치 후 설정 프로필을 선택해 실행합니다.

```bash
ros2 launch synapse synapse.launch.py config:=default
```

주요 launch 인수:

| 인수 | 기본값 | 설명 |
| --- | --- | --- |
| `config` | `default` | `config_main.yaml`에 정의한 프로필 이름 |
| `ui` | `tui` | `tui`(curses 터미널) 또는 `gui`(Qt 창) |
| `debug` | `false` | `true`: 디버그 로그와 debugpy 활성화, `wait`: 디버거 연결까지 대기 |

예:

```bash
ros2 launch synapse synapse.launch.py config:=avp_test ui:=gui
ros2 launch synapse synapse.launch.py config:=gr00t_franka_sim debug:=true
```

실행 중 UI에서 Brain을 선택하고 시작/일시정지하거나, 시나리오 모드로 전환할 수 있습니다. 실제 키/버튼 사용법은 선택한 UI 구현을 참고하세요.

## 설정 파일

패키지 설정은 `src/synapse/configs/`에 있습니다. 설치 후에는 ROS 2 패키지 share 디렉터리의 `synapse/configs/`에서 읽힙니다. 설정을 바꾼 뒤 워크스페이스를 다시 빌드하고 환경을 source하세요.

### 1. `config_main.yaml` — 실행 프로필

맨 위의 `default` 같은 프로필 아래에서 실행 구성을 선택합니다. 보통 다음 값을 수정합니다.

- `embodiment_name`: 사용할 로봇 구성을 `embodiment_configs.yaml`의 키와 일치시킵니다.
- `muscle_option`: 명령 대상 백엔드. 코드상 `DUMMY`, `ISAAC`, `RVIZ`, `REAL_ROBOT` 분기가 있습니다. RVIZ 분기는 실행 파일이 현재 설치 대상에 포함되어 있지 않으므로 사용 전 구현/설치 상태를 확인하세요.
- `brain_registry`: UI에 등록할 Brain의 `노드이름:유형` 목록. 예: `manual_adapter:MANUAL`, `policy:GR00T`.
- 어댑터별 설정 블록: 레지스트리에서 지정한 노드 이름과 동일한 YAML 키를 둡니다. 예를 들어 `policy:GR00T`는 `policy:` 블록을 사용합니다.
- `bt_tick_frequency_hz`, `obs_buffer_window_size`, `publish_rate_hz`, `render_rate_hz`: 실행·관측·화면 갱신 주기.
- `scenario_filename`: Behavior Tree 시나리오를 실행할 때 사용할 YAML 파일명.

GR00T 프로필에서는 정책 서버의 `policy_ip`, `policy_port`, `default_command`를 설정합니다. AVP 프로필에서는 `avp_ip`, 좌우 팔/손 매핑, 좌표계 회전·스케일, 속도 제한 및 손가락 게인을 설정합니다. ROS bag 프로필에서는 기록 파일명, ROS 토픽과 로봇 구성요소 매핑을 지정합니다.

### 2. `embodiment_configs.yaml` — 로봇/실행 대상 정의

`embodiment_name`에 해당하는 항목을 수정합니다. 로봇별로 다음 정보를 정의합니다.

- articulation 및 구성요소(팔, 손 등), 관절 이름·인덱스·한계
- 상태 구독 토픽(`states`)
- 백엔드별 명령 대상(`target`): `ISAAC`, `DUMMY`, `REAL_ROBOT` 등
- 말단 프레임, URDF/USD 파일, 카메라 및 좌표 관련 값

실제 로봇에 연결하려면 상태/명령 토픽과 메시지 형식이 드라이버와 일치해야 합니다. `REAL_ROBOT` 설정은 실제 장비를 움직일 수 있으므로 토픽, 관절 순서, 속도/이동 시간 값을 확인한 뒤 사용하세요.

### 3. 시나리오 YAML — 동작 절차

`scenario_filename`으로 선택하는 `scn_*.yaml` 파일에서 Behavior Tree 동작을 정의합니다. 새 시나리오를 추가한다면 `src/synapse/configs/` 아래에 저장하고 프로필의 파일명을 맞추세요. 시나리오 파서가 기대하는 필드와 동작 형식은 `synapse/utils/scenario_parser.py` 및 기존 `scn_v2.yaml`, `scn_mover_test.yaml` 예시를 따릅니다.

## 현재 포함된 Brain 어댑터

`src/synapse/synapse/brains/brain_selector.py`의 등록 목록 기준입니다.

| 유형 | 용도 |
| --- | --- |
| `MANUAL` | UI 기반 수동 조작 |
| `GR00T` | 원격 GR00T/VLA 정책 서버와 통신 |
| `RANDOM` | 임의 액션 생성, 파이프라인 시험 |
| `DUMMY` | 더미 액션/테스트용 Brain |
| `ROSBAG` | JSON 형식의 기록된 ROS 액션 재생 |
| `MOVE` | 시나리오에서 지정한 목표로 관절/말단 동작 생성 |
| `AVP` | AVP Tracking Streamer 입력을 로봇 조작으로 변환 |

프로필에서 각 어댑터를 `노드이름:유형` 형식으로 등록합니다. 같은 어댑터 유형을 서로 다른 노드 이름과 설정 블록으로 여러 번 등록할 수 있습니다(예: 서로 다른 명령/포트를 가진 GR00T 정책 두 개).

## Muscle 백엔드

launch 파일은 `muscle_option`에 따라 `DUMMY`, `ISAAC`, `REAL_ROBOT` 노드를 선택합니다. 소스에는 `RVIZ` 분기도 있으나 현재 CMake 설치 목록에는 `rviz_node.py`가 없습니다. 각 Embodiment 구성요소의 `target`도 선택한 백엔드에 맞춰야 합니다.

## 확장하기

새 Brain 어댑터는 기존 어댑터의 공통 인터페이스를 구현한 뒤 `brain_selector.py`의 `BRAINS` 맵에 유형과 클래스를 등록하고, `config_main.yaml`의 `brain_registry`에 추가합니다. 관측/액션 형식은 `base_brain_adapter.py`와 기존 어댑터 구현을 기준으로 맞추세요.

## 주요 디렉터리

- `src/synapse/configs/`: 프로필, 로봇 Embodiment, 시나리오 YAML
- `src/synapse/launch/`: ROS 2 launch 파일
- `src/synapse/synapse/brains/adapters/`: Brain 어댑터
- `src/synapse/synapse/muscles/`: DUMMY, Isaac Sim, 실제 로봇 노드
- `src/synapse/synapse/ui/`: curses TUI, Qt GUI
- `src/synapse/synapse/utils/`: 설정/시나리오 파서, 기구학, 궤적 생성 도구
