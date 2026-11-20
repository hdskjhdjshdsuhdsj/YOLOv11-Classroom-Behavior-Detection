@echo off
chcp 936 >nul
cd /d %~dp0
set PY=C://ProgramData//anaconda3//python.exe
set PATH=C://ProgramData//anaconda3//Library//bin;%PATH%

echo ============================================
echo  YOLO课堂行为检测 + DeepSeek AI 分析
echo ============================================
echo.

if "%~1"=="" (
  echo 用法: 把图片拖到本bat文件上即可分析
  echo 或命令行: 运行分析.bat 图片路径
  echo.
  echo 当前为演示模式, 按任意键继续...
  pause >nul
  "%PY%" main.py --mock --save
) else (
  echo 检测图片: %~1
  "%PY%" main.py --image "%~1" --save
)

echo.
pause
