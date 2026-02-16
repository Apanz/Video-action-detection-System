#!/usr/bin/env python3
"""
视频动作识别的实时检测脚本
使用摄像头或视频文件进行实时动作检测的命令行界面
"""

import sys
import argparse
from pathlib import Path

# 将src目录添加到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from detection import DetectionPipeline


def main():
    """主检测函数"""
    parser = argparse.ArgumentParser(
        description='Real-time video action detection using TSN and YOLO'
    )

    # 输入模式
    parser.add_argument('--mode', type=str, default='webcam',
                        choices=['webcam', 'video'],
                        help='Detection mode: webcam or video file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to video file (required for video mode)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index for webcam mode (default: 0)')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained TSN model checkpoint')
    parser.add_argument('--yolo_model', type=str, default='yolov5s',
                        choices=['yolov5s', 'yolov8n', 'yolov8s', 'yolov8m'],
                        help='YOLO model for person detection')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold (0.0-1.0)')

    # 输出参数
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Output video FPS')
    parser.add_argument('--no_display', action='store_true',
                        help='Disable display window (headless mode)')

    args = parser.parse_args()

    # 验证参数
    if args.mode == 'video' and args.input is None:
        print("Error: --input required for video mode")
        parser.print_help()
        sys.exit(1)

    # 验证检查点是否存在
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # 打印配置
    print("="*60)
    print("Real-time Action Detection")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.mode == 'webcam':
        print(f"Camera Index: {args.camera}")
    else:
        print(f"Input Video: {args.input}")
    print(f"TSN Checkpoint: {args.checkpoint}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"Detection Confidence: {args.confidence}")
    if args.output:
        print(f"Output Video: {args.output}")
    print(f"Display: {'Disabled' if args.no_display else 'Enabled'}")
    print("="*60)

    try:
        # 创建检测流水线
        print("\nInitializing detection pipeline...")
        pipeline = DetectionPipeline(
            checkpoint_path=args.checkpoint,
            yolo_model=args.yolo_model,
            output_path=args.output,
            fps=args.fps,
            show_display=not args.no_display,
            save_video=args.output is not None
        )

        # 运行检测
        print("Starting detection...")
        print("Press 'q' to quit")

        if args.mode == 'webcam':
            stats = pipeline.process_webcam(camera_index=args.camera)
        else:
            stats = pipeline.process_video(video_path=args.input, output_path=args.output)

        # 打印最终统计
        print("\n" + "="*60)
        print("Detection Complete")
        print("="*60)
        print(f"Total frames processed: {stats.get('frames_processed', 0)}")
        print(f"Total detections: {stats.get('detections', 0)}")
        print(f"Total classifications: {stats.get('classifications', 0)}")
        if 'average_fps' in stats:
            print(f"Average FPS: {stats['average_fps']:.1f}")
        if 'total_time' in stats:
            print(f"Total time: {stats['total_time']:.1f}s")
        print("="*60)

        # 清理
        pipeline.close()

    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
        pipeline.close()
        sys.exit(0)
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
