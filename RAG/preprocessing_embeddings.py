import json

input_filename = "transformed_with_embeddings_second.json"
output_filename = "transformed_with_embeddings_without_related_links_second.json"

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"'{input_filename}' 파일을 성공적으로 로드했습니다.")

    deleted_count = 0
    # 데이터가 리스트인지 확인
    if isinstance(data, list):
        # 리스트의 모든 항목을 순회
        for index, item in enumerate(data):
            # 항목이 딕셔너리이고 'metadata' 키를 가지고 있는지 확인
            if isinstance(item, dict) and 'metadata' in item:
                # 'metadata' 값이 딕셔너리이고 'related_links' 키를 가지고 있는지 확인
                if isinstance(item['metadata'], dict) and 'related_links' in item['metadata']:
                    del item['metadata']['related_links']
                    print(f"리스트 {index}번째 항목의 'metadata' 내부 'related_links' 키를 삭제했습니다.")
                    deleted_count += 1
                # else: # 필요하다면 related_links가 없는 경우 메시지 출력
                #    print(f"리스트 {index}번째 항목의 'metadata' 내부에 'related_links'가 없습니다.")
            # else: # 필요하다면 metadata 키가 없는 경우 메시지 출력
            #    print(f"리스트 {index}번째 항목에 'metadata' 키가 없습니다.")
        if deleted_count == 0:
             print("삭제할 'related_links' 키를 가진 항목을 찾지 못했습니다.")

    else:
        print("경고: 로드된 데이터가 리스트 형태가 아닙니다.")

    # 수정된 데이터를 새 JSON 파일로 저장 (이하 동일)
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"수정된 데이터가 '{output_filename}' 파일로 성공적으로 저장되었습니다.")
    except IOError as e:
        print(f"'{output_filename}' 파일 저장 중 오류 발생: {e}")

except FileNotFoundError:
    print(f"오류: 입력 파일 '{input_filename}'을 찾을 수 없습니다.")
