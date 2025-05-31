import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import deque, defaultdict
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import signal

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

# =============================================================================
# 환경 관리
# =============================================================================
class EnvironmentManager:
    """MORAI 환경 관리 클래스"""
    
    @staticmethod
    def setup_environment(action_bounds):
        """환경 초기화"""
        try:
            from gym_morai.envs.morai_env import MoraiEnv
            from gym_morai.envs.reward_fns import RewardFns
            from gym_morai.envs.terminated_fns import TerminatedFns
        except ImportError:
            raise ImportError("MORAI environment is required")
        
        env = MoraiEnv(action_bounds=action_bounds)
        sensor = env.sensor
        
        # 보상 및 종료 함수 설정
        env.set_reward_fn(RewardFns.adaptive_speed_lanefollow_reward(sensor))
        env.set_episode_over_fn(TerminatedFns.cte_done(sensor, max_cte=0.5))
        
        return env, sensor
    
    @staticmethod
    def force_reset_environment(env, action_bounds):
        """환경 강제 재초기화"""
        try:
            print("[RESET] 환경 강제 재초기화 중...")
            env.close()
            time.sleep(2)
            
            new_env, new_sensor = EnvironmentManager.setup_environment(action_bounds)
            print("[RESET] 환경 재초기화 완료")
            return new_env, new_sensor
        except Exception as e:
            print(f"[ERROR] 환경 재초기화 실패: {e}")
            return env, env.sensor

class EnvironmentLabelManager:
    """환경 라벨 관리 클래스"""
    
    def __init__(self, manual_env=None):
        self.manual_env = manual_env
        self.env_name_to_label = {'solid': 0, 'dashed': 1, 'dash': 1, 'shadow': 2}
        self.env_label_to_name = {0: 'solid', 1: 'dashed', 2: 'shadow'}
        
        # 환경 감지기는 auto 모드에서만 사용
        if manual_env is None:
            try:
                from Multi_PPO import LaneEnvironmentDetector
                self.env_detector = LaneEnvironmentDetector()
            except ImportError:
                self.env_detector = None
        else:
            self.env_detector = None
        
    def get_environment_label(self, obs_dict, sensor=None, episode=None, step=None):
        """환경 라벨 획득"""
        if self.manual_env is not None:
            # 수동 지정 모드
            return self.env_name_to_label.get(self.manual_env.lower(), 0)
        elif self.env_detector is not None:
            # 자동 감지 모드
            return self.env_detector.detect_lane_environment(obs_dict, sensor)
        else:
            # 기본값
            return 0
    
    def get_environment_name(self, env_label):
        """환경 라벨을 이름으로 변환"""
        return self.env_label_to_name.get(env_label, 'unknown')
    
    def print_mode_info(self):
        """현재 환경 모드 정보 출력"""
        if self.manual_env is not None:
            env_label = self.env_name_to_label.get(self.manual_env.lower(), 0)
            print(f"[ENV] 수동 환경 모드: {self.manual_env} (라벨={env_label})")
            print(f"[ENV] 해당 환경의 Critic만 학습됩니다.")
        else:
            print(f"[ENV] 자동 감지 모드: 이미지 분석을 통한 환경 자동 분류")

# =============================================================================
# 학습 통계 관리
# =============================================================================
class TrainingStats:
    """학습 통계 관리 클래스"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.total_short_episodes = 0
        self.total_step1_count = 0
        self.consecutive_short_episodes = 0
        self.env_distribution = {'solid': 0, 'dashed': 0, 'shadow': 0}
        self.env_names = ['solid', 'dashed', 'shadow']
        
    def add_episode(self, reward, length, env_label):
        """에피소드 통계 추가"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.env_distribution[self.env_names[env_label]] += 1
        
    def add_short_episode(self, length):
        """짧은 에피소드 통계 추가"""
        self.consecutive_short_episodes += 1
        self.total_short_episodes += 1
        if length == 1:
            self.total_step1_count += 1
    
    def reset_consecutive_count(self):
        """연속 짧은 에피소드 카운트 리셋"""
        self.consecutive_short_episodes = 0
    
    def is_invalid_episode(self, episode_steps, min_steps=10):
        """유효하지 않은 에피소드 판별"""
        return episode_steps <= min_steps
    
    def print_stats(self, episode, manual_env=None):
        """통계 출력"""
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(self.episode_lengths[-50:]) if len(self.episode_lengths) >= 50 else np.mean(self.episode_lengths)
            
            print(f"[STATS] Episode {episode}: 최근 평균 - Reward: {avg_reward:.2f}, Length: {avg_length:.1f}")
            print(f"[STATS] 실패 에피소드: Step1={self.total_step1_count}, 짧은 에피소드={self.total_short_episodes}")
            
            # 환경 분포 출력
            total_samples = sum(self.env_distribution.values())
            if total_samples > 0:
                for env_name, count in self.env_distribution.items():
                    ratio = count / total_samples
                    print(f"[STATS] {env_name}: {ratio:.2%}")
            
            # 수동 환경 모드 추가 정보
            if manual_env:
                print(f"[FOCUS] 현재 학습 중인 환경: {manual_env}")

    def get_summary(self):
        """학습 완료 후 요약 통계"""
        return {
            'total_episodes': len(self.episode_rewards),
            'success_episodes': len(self.episode_rewards),
            'failed_episodes': self.total_short_episodes,
            'step1_episodes': self.total_step1_count,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'success_rate': len(self.episode_rewards) / (len(self.episode_rewards) + self.total_short_episodes) * 100 if (len(self.episode_rewards) + self.total_short_episodes) > 0 else 0,
            'env_distribution': self.env_distribution.copy()
        }

# =============================================================================
# 학습 세션 관리자
# =============================================================================
class TrainingSession:
    """전체 학습 세션을 관리하는 클래스"""
    
    def __init__(self, manual_env=None, save_dir="/home/kuuve/catkin_ws/src/pt/", log_dir=None):
        self.manual_env = manual_env
        self.save_dir = save_dir
        
        # 로그 디렉토리 설정
        if log_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            env_suffix = f"_{manual_env}" if manual_env else ""
            self.log_dir = f"/home/kuuve/catkin_ws/src/logs/multi_critic{env_suffix}_{timestamp}"
        else:
            self.log_dir = log_dir
        
        # 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 컴포넌트 초기화
        self.env_manager = EnvironmentLabelManager(manual_env)
        self.stats = TrainingStats()
        self.tb_logger = None
        self.performance_analyzer = None
        
        # 종료 플래그 설정
        self.stop_flag = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """키보드 인터럽트 핸들러"""
        self.stop_flag = True
        print("\n학습 중단 신호를 받았습니다. 안전하게 종료 중...")
        
    def setup_logging(self, agent):
        """로깅 시스템 초기화"""
        if TENSORBOARD_AVAILABLE:
            experiment_name = f"multi_critic_ppo"
            if self.manual_env:
                experiment_name += f"_{self.manual_env}"
            
            self.tb_logger = TensorBoardLogger(self.log_dir, experiment_name)
            self.performance_analyzer = PerformanceAnalyzer(self.tb_logger)
            
            # 하이퍼파라미터 로깅
            hparams = agent.get_hyperparameters()
            hparams['manual_environment'] = self.manual_env or 'auto_detection'
            self.tb_logger.log_hyperparameters(hparams)
            
            print(f"📊 TensorBoard: tensorboard --logdir {self.log_dir}")
        else:
            print("⚠️  TensorBoard 미설치 - 기본 로깅만 사용")
    
    def log_training_metrics(self, global_step, train_metrics, manual_env=None):
        """학습 지표 로깅"""
        if self.tb_logger and train_metrics:
            self.tb_logger.log_training_losses(global_step, train_metrics)
            self.tb_logger.log_classifier_metrics(global_step, train_metrics['classifier_accuracy'])
            
            # 환경별 학습 진행 상황 로깅
            if manual_env:
                env_label = self.env_manager.env_name_to_label.get(manual_env.lower(), 0)
                self.tb_logger.log_custom_metric(f'Training/Focus_Environment', env_label, global_step)
                self.tb_logger.log_custom_metric(f'Training/{manual_env}_Critic_Loss', 
                                              train_metrics['critic_loss'], global_step)
    
    def log_episode_metrics(self, episode, reward, length, env_label, manual_env=None):
        """에피소드 지표 로깅"""
        if self.tb_logger:
            self.tb_logger.log_episode_metrics(episode, reward, length)
            
            # 환경별 성능 로깅
            env_name = self.env_manager.get_environment_name(env_label)
            self.tb_logger.log_custom_metric(f'Performance/Reward_{env_name}', reward, episode)
            
            if manual_env:
                self.tb_logger.log_custom_metric(f'Performance/Reward_Focus_{manual_env}', reward, episode)
        
        # 성능 분석 데이터 추가
        if self.performance_analyzer:
            episode_env_dist = {name: 0 for name in self.stats.env_names}
            episode_env_dist[self.stats.env_names[env_label]] = 1
            self.performance_analyzer.add_episode_data(episode, reward, length, episode_env_dist, 0.0)
    
    def should_stop(self):
        """학습 중단 여부 확인"""
        return self.stop_flag
    
    def get_save_path(self, final=False):
        """모델 저장 경로 반환"""
        if self.manual_env:
            suffix = "final_model" if final else "model"
            return os.path.join(self.save_dir, f"{suffix}_{self.manual_env}")
        else:
            return self.save_dir
    
    def analyze_performance(self):
        """성능 분석 실행"""
        if self.performance_analyzer:
            self.performance_analyzer.analyze_convergence()
            self.performance_analyzer.analyze_environment_adaptation()
    
    def print_experiment_info(self):
        """실험 정보 출력"""
        print("Multi-Critic PPO 강화학습 시작")
        print("=" * 60)
        
        if self.manual_env:
            print(f"🎯 환경 지정 모드: {self.manual_env.upper()}")
            env_label = self.env_manager.env_name_to_label.get(self.manual_env.lower(), 0)
            print(f"   → Critic_{env_label} 전용 학습")
            print(f"   → 다른 환경의 Critic은 업데이트되지 않습니다")
        else:
            print(f"🔍 자동 감지 모드: 이미지 분석을 통한 환경 자동 분류")
            print(f"   → 모든 Critic이 상황에 따라 학습됩니다")
        
        print("=" * 60)
    
    def print_final_summary(self, episode_count):
        """최종 학습 결과 요약"""
        summary = self.stats.get_summary()
        
        print("\nMulti-Critic PPO 학습 완료!")
        print(f"총 에피소드: {episode_count}")
        print(f"정상 에피소드: {summary['success_episodes']}개")
        print(f"실패 에피소드: {summary['failed_episodes']}개 (Step1: {summary['step1_episodes']})")
        print(f"최종 평균 보상: {summary['final_avg_reward']:.2f}")
        print(f"성공률: {summary['success_rate']:.1f}%")
        
        if self.manual_env:
            env_label = self.env_manager.env_name_to_label.get(self.manual_env.lower(), 0)
            print(f"[FOCUS] 학습된 환경: {self.manual_env} (Critic_{env_label})")
            print(f"모델 저장 위치: {self.get_save_path(final=True)}")
        
        if self.tb_logger:
            print(f"📊 TensorBoard: tensorboard --logdir {self.log_dir}")
    
    def cleanup(self, env):
        """리소스 정리"""
        if self.tb_logger:
            self.tb_logger.close()
        if env:
            env.close()

# =============================================================================
# 기존 유틸리티 클래스들
# =============================================================================
class Cal_CTE:
    @staticmethod
    def load_centerline(csv_path):
        x_list = []
        y_list = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # 헤더 스킵
            for row in reader:
                x = float(row[1])  # 2열: x
                y = float(row[2])  # 3열: y
                x_list.append(x)
                y_list.append(y)
        return np.array(list(zip(x_list, y_list)))  # (N, 2) array

    @staticmethod
    def calculate_cte(agent_pos, path_points):
        agent_pos = np.array(agent_pos)  # tuple -> ndarray
        distances = np.linalg.norm(path_points - agent_pos, axis=1)
        min_index = np.argmin(distances)
        closest_point = path_points[min_index]
        cte = np.linalg.norm(agent_pos - closest_point)
        return cte

class ConfigManager:
    """설정 관리 유틸리티"""
    
    @staticmethod
    def save_config(config_dict: Dict, save_path: str):
        """설정을 JSON 파일로 저장"""
        # Convert any non-serializable objects to strings
        serializable_config = {}
        for key, value in config_dict.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_config[key] = value
            except TypeError:
                serializable_config[key] = str(value)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_config, f, indent=4)
    
    @staticmethod
    def load_config(load_path: str) -> Dict:
        """JSON 파일에서 설정 로드"""
        with open(load_path, 'r') as f:
            return json.load(f)

class TensorBoardLogger:
    """
    Multi-Critic PPO를 위한 TensorBoard 로깅 클래스
    논문의 평가 지표들을 체계적으로 기록
    """
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not available. Install with: pip install tensorboard")
            
        if experiment_name is None:
            experiment_name = f"multi_critic_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.experiment_name = experiment_name
        
        # 이동 평균 계산을 위한 버퍼
        self.reward_buffer = deque(maxlen=100)
        self.step_counter = 0
        self.episode_counter = 0
        
        # Critic 선택 오류율 추적
        self.critic_selection_errors = deque(maxlen=1000)
        
    def log_episode_metrics(self, episode: int, episode_reward: float, episode_length: int = None):
        """에피소드별 지표 로깅"""
        self.episode_counter = episode
        self.reward_buffer.append(episode_reward)
        
        # Reward/Episode - 정책의 전체 성능 추이 확인
        self.writer.add_scalar('Reward/Episode', episode_reward, episode)
        
        # Reward/MovingAverage - 최근 에피소드 평균 보상 (시각적 추세 확인용)
        if len(self.reward_buffer) >= 10:  # 최소 10개 에피소드 후부터
            moving_avg = np.mean(list(self.reward_buffer))
            self.writer.add_scalar('Reward/MovingAverage', moving_avg, episode)
        
        # Episode length tracking
        if episode_length is not None:
            self.writer.add_scalar('Episode/Length', episode_length, episode)
            
    def log_training_losses(self, step: int, loss_dict: Dict[str, float]):
        """학습 과정의 손실 함수들 로깅"""
        self.step_counter = step
        
        # Loss/Actor - actor(policy)의 학습 수렴 추적
        if 'actor_loss' in loss_dict:
            self.writer.add_scalar('Loss/Actor', loss_dict['actor_loss'], step)
        
        # Loss/Critic_{env} - critic의 환경별 학습 상태 추적
        if 'critic_losses' in loss_dict:
            for env_name, critic_loss in loss_dict['critic_losses'].items():
                self.writer.add_scalar(f'Loss/Critic_{env_name}', critic_loss, step)
        
        # Total critic loss
        if 'critic_loss' in loss_dict:
            self.writer.add_scalar('Loss/Critic_Total', loss_dict['critic_loss'], step)
        
        # Loss/Classifier - 환경 분류기의 학습 수렴 확인
        if 'classifier_loss' in loss_dict:
            self.writer.add_scalar('Loss/Classifier', loss_dict['classifier_loss'], step)
        
        # Total loss
        if 'total_loss' in loss_dict:
            self.writer.add_scalar('Loss/Total', loss_dict['total_loss'], step)
            
        # Entropy loss
        if 'entropy' in loss_dict:
            self.writer.add_scalar('Loss/Entropy', loss_dict['entropy'], step)
    
    def log_classifier_metrics(self, step: int, accuracy: float, predictions: np.ndarray = None, 
                             ground_truth: np.ndarray = None):
        """분류기 성능 지표 로깅"""
        
        # Accuracy/Classifier - 환경 분류기 정확도 평가 (validation 기준)
        self.writer.add_scalar('Accuracy/Classifier', accuracy, step)
        
        # 혼동 행렬 및 클래스별 정확도
        if predictions is not None and ground_truth is not None:
            # 클래스별 정확도
            for class_idx in range(3):  # solid, dashed, shadow
                class_mask = (ground_truth == class_idx)
                if np.any(class_mask):
                    class_accuracy = np.mean(predictions[class_mask] == ground_truth[class_mask])
                    class_names = ['solid', 'dashed', 'shadow']
                    self.writer.add_scalar(f'Accuracy/Classifier_{class_names[class_idx]}', 
                                         class_accuracy, step)
    
    def log_custom_metric(self, name: str, value: float, step: int):
        """사용자 정의 지표 로깅"""
        self.writer.add_scalar(name, value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, any], metrics: Dict[str, float] = None):
        """하이퍼파라미터 로깅"""
        if metrics is None:
            metrics = {'placeholder': 0.0}  # TensorBoard requires at least one metric
            
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """TensorBoard writer 종료"""
        if hasattr(self, 'writer'):
            self.writer.close()

class PerformanceAnalyzer:
    """성능 분석을 위한 고급 유틸리티"""
    
    def __init__(self, tb_logger: TensorBoardLogger):
        self.tb_logger = tb_logger
        self.episode_data = []
        
    def add_episode_data(self, episode: int, reward: float, length: int, 
                        env_distribution: Dict[str, int], avg_cte: float):
        """에피소드 데이터 수집"""
        self.episode_data.append({
            'episode': episode,
            'reward': reward,
            'length': length,
            'env_distribution': env_distribution,
            'avg_cte': avg_cte
        })
    
    def analyze_convergence(self, window_size: int = 100):
        """수렴성 분석"""
        if len(self.episode_data) < window_size:
            return
            
        recent_rewards = [ep['reward'] for ep in self.episode_data[-window_size:]]
        
        # 분산 분석
        reward_variance = np.var(recent_rewards)
        self.tb_logger.log_custom_metric('Analysis/RewardVariance', reward_variance, 
                                       self.episode_data[-1]['episode'])
        
        # 트렌드 분석
        x = np.arange(len(recent_rewards))
        slope, _, r_value, _, _ = stats.linregress(x, recent_rewards)
        self.tb_logger.log_custom_metric('Analysis/RewardTrend', slope, 
                                       self.episode_data[-1]['episode'])
        self.tb_logger.log_custom_metric('Analysis/TrendCorrelation', r_value, 
                                       self.episode_data[-1]['episode'])
    
    def analyze_environment_adaptation(self):
        """환경별 적응성 분석"""
        if len(self.episode_data) < 50:
            return
            
        # 환경별 성능 분석
        env_performance = {'solid': [], 'dashed': [], 'shadow': []}
        
        for ep_data in self.episode_data[-50:]:  # 최근 50 에피소드
            total_samples = sum(ep_data['env_distribution'].values())
            if total_samples > 0:
                for env_name in env_performance.keys():
                    env_ratio = ep_data['env_distribution'].get(env_name, 0) / total_samples
                    if env_ratio > 0.3:  # 해당 환경이 30% 이상인 에피소드만
                        env_performance[env_name].append(ep_data['reward'])
        
        # 환경별 평균 성능 로깅
        episode = self.episode_data[-1]['episode']
        for env_name, rewards in env_performance.items():
            if rewards:
                avg_performance = np.mean(rewards)
                self.tb_logger.log_custom_metric(f'Analysis/Performance_{env_name}', 
                                               avg_performance, episode)

# =============================================================================
# 기타 유틸리티
# =============================================================================
class Plot:
    @staticmethod
    def save_reward_csv(reward_list, csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])
            for i, r in enumerate(reward_list, start=1):
                writer.writerow([i, r])

    @staticmethod
    def plot_rewards(reward_list, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(reward_list, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Reward Curve")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()