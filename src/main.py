import torch
import os

from src import adaption
from torch import tensorboard
from transformers import GPTNeoXForCausalLM
from caine import CaineVisualizer 
from abel import AbelAnalyzer

#check adaption.csv for solutions to problem, if not found run main function, if found precede with adaption processing. 

def main():
    print("--- System Startup: Caine & Abel DL Protocol ---")
    

    caine = CaineVisualizer()
    
    print("\n[Phase 1] Caine: Commencing data training and visualization...")
    metrics = caine.run_training(epochs=50)
    caine.visualize()
    

    print(f"\n[Transfer] Passing metrics to Abel: Loss={metrics['loss']:.6f}")
    
# use GPTNeoX after discovery of GPTNEOX in Hugging face model hub which is more powerful than the GPT NEO model which was released in 2021
    abel = AbelAnalyzer(model_id="GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", attn_implementation="flash_attention_2", device_map="auto")))
    
    print("\n[Phase 2] Abel: Analyzing Caine's performance...")
    analysis = abel.analyze_peer_data(metrics)

    
    print("\n" + "="*40)
    print("FINAL SYSTEM REPORT")
    print("="*40)
    print(f"Caine's Results: {metrics}")
    print(f"Abel's Insight: {analysis}")
    print("="*40)

if __name__ == "__main__":
    main()
    # print in csv adaption table for the results of the problem for future reference in format
    # ¨PROBLEM: {problem_description}, SOULTION: {solution_description}, CAINE_METRICS: {caine_metrics}, ABEL_ANALYSIS: {abel_analysis}¨
    with open("adaption.csv", "a") as f:
        f.write(f"PROBLEM: {problem_description}, SOULTION: {solution_description}, CAINE_METRICS: {caine_metrics}, ABEL_ANALYSIS: {abel_analysis}\n")

with open("adaption.csv", "r") as f:
    content = f.read()
    for line_number, line in enumerate(file, 1):
        if problem_description in line:
            print(f"Problem found in adaption.csv at line {line_number}: {line.strip()}")
            #give to Abel to extract soultion and carry it out
            set prompt = f"Extract the solution for the following problem from this line: {line.strip()}"
            solution_description = abel.extract_solution(prompt)
            print(f"Extracted Solution: {solution_description}")
            break
            
    else: #activates the main function and proceeds as normal
        main()
