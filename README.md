# Dacon

1. 개요 
    - `Dacon` 1월 월간 공모전 
    - `주제명` : 난독화 된 한글 리뷰 데이터 복원 AI 경진대회 

2. `BASE MODEL` 
    - `KETI-AIR/ke-t5-base`

        - 결과 
        ```javascript 
        {'train_runtime': 9027.9662, 'train_samples_per_second': 6.238, 'train_steps_per_second': 0.26, 'train_loss': 46.53049561718125, 'epoch': 4.99}
        ```
    
    - 결과가 좋지 않았음 

> 데이터 생성 결과 
![alt text](./image/image.png)

매우 좋지 않았음....

3. `Try → gemma-2-2b-it` 도전 예정 

![alt text](./image/gemma.png)

- 2025.01.21일 1차 파인튜닝 실험 
    - batch_size = 1, `peft`, `trl` 활용한 양자화 도전(배치를 늘릴수록 VRAM 터지는 사고가...)
    
    - `수정사항` 
        - `batch_size` : 10 
        - `data_size`  : 10000
        - `loss` : 3.8 
    
- `huggingFace link` 
    - https://huggingface.co/UICHEOL-HWANG/Dacon-contest-obfuscation-gemma2-2b

    ![alt text](./image/hug.png)