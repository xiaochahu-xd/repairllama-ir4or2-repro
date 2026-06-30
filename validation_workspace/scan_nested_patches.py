import json

input_jsonl = "../results/preds/concurrency_bugs_qlora4bit_dynamic.jsonl"
print("\n🚀 开始全量扫描【嵌套结构】并发修复成果 🚀\n" + "="*70)

CONCURRENCY_KEYWORDS = ["synchronized", "Lock", "lock", "unlock", "Atomic", "Concurrent", "volatile", "Thread", "sleep"]

match_count = 0
total_count = 0

with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        total_count += 1
        data = json.loads(line.strip())
        bug_id = data.get("bug_id", f"Bug_{total_count}")
        buggy = data.get("buggy_code", "")
        gold = data.get("gold_patch", "").strip()
        
        # 获取嵌套的 output 字典
        outputs = data.get("output", {})
        
        # 遍历模型生成的 10 个候选补丁
        found_in_candidates = False
        print_buffer = []
        
        for cand_id, cand_data in outputs.items():
            patch_code = cand_data.get("output_patch", "")
            
            # 检查是否有并发原语关键字
            if any(kw in patch_code for kw in CONCURRENCY_KEYWORDS):
                found_in_candidates = True
                print_buffer.append(f"   [候选补丁 {cand_id}]: {patch_code.strip()}")
        
        if found_in_candidates:
            match_count += 1
            print(f"\n🔥 【高价值案例 #{match_count}】 Bug ID: {bug_id}")
            print("❌ 【原代码 (存在并发缺陷)】:")
            print("\n".join(buggy.splitlines()[:4]) + "\n...")
            print(f"🎯 【标准答案 (Gold Patch)】: {gold}")
            print("✅ 【模型生成的有效并发候选】:")
            for buf in print_buffer[:4]: # 每一个 Bug 最多展示 4 个候选，防止刷屏
                print(buf)
            print("-" * 70)
            
            if match_count >= 10:
                print("\n⚠️ 已为你精选前 10 个典型案例，其余已隐藏。")
                break

print(f"\n📊 扫描完毕：当前共分析 {total_count} 条数据，成功拦截出 {match_count} 个包含并发控制机制（锁、线程控制）的高价值修复样本！")
