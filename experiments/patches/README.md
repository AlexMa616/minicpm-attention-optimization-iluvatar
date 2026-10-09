# 未验证源码补丁

`2026-10-09-native-decode-unvalidated.patch` 是本地开发HEAD `edcb128` 上
两处未提交修改的原样git diff，涉及独立Decode开关和B8精确grid。
通过过kernel-only测试，但无稳定服务增益验收，不属于正式候选更新。
**不自动apply、不作为安装步骤。**

补丁不含待研究的 `k_descale/v_descale` 条件修复；还需要真实guard、
scale语义审查及新环境数值验收。
