#!/usr/bin/env python3
"""
TensorRT를 위한 완전히 정적인 크기 고정 CNN
"""
import torch
import torch.nn as nn
from torch2trt import torch2trt
import os

class StaticSizeCNN(nn.Module):
    """TensorRT를 위한 완전히 정적인 크기의 CNN"""
    def __init__(self):
        super().__init__()
        
        # ROI 제거하고 고정 크기 입력 사용
        self.backbone = nn.Sequential(
            # 입력: 84x160 (ROI 적용된 크기)
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 42x80
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 21x40
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 11x20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 고정: 4x4
        )
        
        # 완전히 고정된 크기: 128 * 4 * 4 = 2048
        self.feature_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ROI를 미리 적용된 입력을 받음 (84x160)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # 정확히 2048 크기
        x = self.feature_extractor(x)
        return x

class ROIWrapper(nn.Module):
    """ROI 적용 + StaticSizeCNN 래퍼"""
    def __init__(self, static_cnn):
        super().__init__()
        self.roi_crop_ratio = 0.3
        self.static_cnn = static_cnn

    def apply_roi(self, x):
        crop_height = int(x.size(2) * self.roi_crop_ratio)
        return x[:, :, crop_height:, :]

    def forward(self, x):
        # ROI 적용: 120x160 → 84x160
        x_roi = self.apply_roi(x)
        # 정적 CNN 적용
        return self.static_cnn(x_roi)

def create_compatible_weights(original_weights):
    """기존 가중치를 새 모델에 맞게 변환"""
    new_weights = {}
    
    # Backbone 가중치 그대로 복사
    for key, value in original_weights.items():
        if key.startswith('backbone.'):
            new_weights[key] = value
        elif key.startswith('feature_extractor.'):
            new_weights[key] = value
    
    return new_weights

def test_size_compatibility():
    """크기 호환성 테스트"""
    print("=== 크기 호환성 테스트 ===")
    
    # 정적 모델 테스트
    static_model = StaticSizeCNN()
    roi_input = torch.randn(1, 1, 84, 160)  # ROI 적용된 크기
    
    with torch.no_grad():
        output = static_model(roi_input)
        print(f"정적 모델 출력: {output.shape}")
    
    # 래퍼 모델 테스트
    wrapper_model = ROIWrapper(static_model)
    full_input = torch.randn(1, 1, 120, 160)  # 전체 크기
    
    with torch.no_grad():
        output2 = wrapper_model(full_input)
        print(f"래퍼 모델 출력: {output2.shape}")

def convert_static_model():
    """정적 모델로 변환"""
    print("=== 정적 모델 TensorRT 변환 ===")
    
    try:
        # 1. 정적 모델 생성
        static_model = StaticSizeCNN().cuda()
        
        # 2. 기존 가중치 로드
        original_weights = torch.load("/home/kuuve/catkin_ws/src/pt/PPO_encoder.pt", weights_only=True)
        compatible_weights = create_compatible_weights(original_weights)
        static_model.load_state_dict(compatible_weights)
        static_model.eval()
        print("✓ 정적 모델 가중치 로드 완료")
        
        # 3. ROI 적용된 크기로 테스트 (84x160)
        roi_input = torch.randn(1, 1, 84, 160).cuda()
        
        with torch.no_grad():
            output = static_model(roi_input)
            print(f"✓ 정적 모델 테스트: {output.shape}")
        
        # 4. TensorRT 변환 (ROI 적용된 크기로)
        print("TensorRT 변환 중 (84x160 입력)...")
        static_trt = torch2trt(static_model, [roi_input])
        
        # 5. 변환 확인
        with torch.no_grad():
            trt_output = static_trt(roi_input)
            diff = torch.abs(output - trt_output).max()
            print(f"✓ 정적 모델 변환 성공, 차이: {diff:.6f}")
        
        # 6. 정적 모델 저장
        os.makedirs("trt_models", exist_ok=True)
        torch.save(static_trt.state_dict(), "trt_models/PPO_encoder_static_trt.pth")
        print("✓ 정적 모델 저장 완료")
        
        # 7. 전체 래퍼 모델 변환 (120x160 입력)
        print("\n전체 래퍼 모델 변환 중 (120x160 입력)...")
        wrapper_model = ROIWrapper(static_model).cuda()
        full_input = torch.randn(1, 1, 120, 160).cuda()
        
        wrapper_trt = torch2trt(wrapper_model, [full_input])
        
        with torch.no_grad():
            original_output = wrapper_model(full_input)
            wrapper_trt_output = wrapper_trt(full_input)
            wrapper_diff = torch.abs(original_output - wrapper_trt_output).max()
            print(f"✓ 래퍼 모델 변환 성공, 차이: {wrapper_diff:.6f}")
        
        torch.save(wrapper_trt.state_dict(), "trt_models/PPO_encoder_trt.pth")
        print("✓ 최종 모델 저장 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_actor_critic():
    """Actor와 Critic 변환"""
    print("\n=== Actor/Critic 변환 ===")
    
    import sys
    sys.path.append('/home/kuuve/catkin_ws/src')
    from models.PPO import PPOActor, PPOCritic
    
    try:
        # Actor
        actor = PPOActor(258, 2).cuda()
        actor.load_state_dict(torch.load("/home/kuuve/catkin_ws/src/pt/PPO_actor.pt", weights_only=True))
        actor.eval()
        
        actor_input = torch.randn(1, 258).cuda()
        actor_trt = torch2trt(actor, [actor_input])
        torch.save(actor_trt.state_dict(), "trt_models/PPO_actor_trt.pth")
        print("✓ Actor 변환 완료")
        
        # Critic
        critic = PPOCritic(258).cuda()
        critic.load_state_dict(torch.load("/home/kuuve/catkin_ws/src/pt/PPO_critic.pt", weights_only=True))
        critic.eval()
        
        critic_input = torch.randn(1, 258).cuda()
        critic_trt = torch2trt(critic, [critic_input])
        torch.save(critic_trt.state_dict(), "trt_models/PPO_critic_trt.pth")
        print("✓ Critic 변환 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ Actor/Critic 변환 실패: {e}")
        return False

def main():
    print("정적 크기 고정 TensorRT 변환 시작...")
    
    # 크기 호환성 테스트
    test_size_compatibility()
    
    # 변환 실행
    encoder_success = convert_static_model()
    
    if encoder_success:
        actor_critic_success = convert_actor_critic()
        
        if actor_critic_success:
            print("\n🎉 모든 변환 완료!")
            print("생성된 파일들:")
            print("  - trt_models/PPO_encoder_trt.pth (전체 모델)")
            print("  - trt_models/PPO_encoder_static_trt.pth (정적 모델)")
            print("  - trt_models/PPO_actor_trt.pth")
            print("  - trt_models/PPO_critic_trt.pth")
        else:
            print("\n⚠️ Encoder만 변환 완료")
    else:
        print("\n❌ 모든 변환 실패")

if __name__ == "__main__":
    main()