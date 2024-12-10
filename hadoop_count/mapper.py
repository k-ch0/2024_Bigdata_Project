#!/usr/bin/env python3

import sys
import csv

# 한글 및 특수기호 처리를 위해 UTF-8로 읽기
def process_line(line):
    try:
        line = line.strip()
        reader = csv.reader([line], delimiter=',', quotechar='"')
        for row in reader:
            # 필요한 키와 값을 처리 (예: 첫 번째 컬럼과 두 번째 컬럼)
            key = row[0]  # 예: 첫 번째 컬럼
            value = row[1]  # 예: 두 번째 컬럼
            print(f"{key}\t{value}")
    except Exception as e:
        sys.stderr.write(f"Error processing line: {line}\n{e}\n")

# 표준 입력에서 데이터 읽기
for line in sys.stdin:
    process_line(line)
