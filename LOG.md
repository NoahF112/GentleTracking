### 20260316
Changed y1 armature for tuning PD parameters

**To "soft" impact response: allow a little oscillation and lower response**
- ARMATURE_FOR_PD_HIP_PITCH: 0.5 * 0.031752 -> 0.25 * 0.031752
- ARMATURE_FOR_PD_KNEE: 0.5 * 0.065 -> 0.25 * 0.065

****
- ARMATURE_FOR_PD_SHOULDER_PITCH: 0.5 * 0.032 -> 0.032
- ARMATURE_FOR_PD_SHOULDER_ROLL: 0.5 * 0.032 -> 0.032


- ARMATURE_FOR_PD_ANKLE_PITCH: 0.5 * 0.023328 -> 0.023328
- ARMATURE_FOR_PD_ANKLE_ROLL = 0.5 * 0.023328 -> 0.023328
