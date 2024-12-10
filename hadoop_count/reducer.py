#!/usr/bin/env python3

import sys
from collections import defaultdict

# 키별로 값을 집계하기 위한 딕셔너리
aggregated_data = defaultdict(list)

# 표준 입력에서 데이터 읽기
for line in sys.stdin:
    try:
        line = line.strip()
        key, value = line.split("\t")
        aggregated_data[key].append(value)
    except Exception as e:
        sys.stderr.write(f"Error processing line: {line}\n{e}\n")

# 집계 결과 출력
for key, values in aggregated_data.items():
    # 예: 키별 값의 개수를 출력
    print(f"{key}\t{len(values)}")
