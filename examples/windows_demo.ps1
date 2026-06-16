hippo init
hippo write --project glasses --type constraint --content "用户不接受 3-4 cm 焦距的光学结构。"
hippo write --project glasses --type task_state --content "当前目标是先让 STM32 点亮 TFT 屏幕。"
hippo search "继续上次那个屏幕项目" --project glasses
hippo pack "继续上次那个 STM32 点亮 TFT 的项目" --project glasses
hippo project-profile --project glasses
hippo impact "继续调试 STM32 点亮 TFT 屏幕" --project glasses
hippo run --project glasses --intent "继续调试 STM32 点亮 TFT 屏幕"
