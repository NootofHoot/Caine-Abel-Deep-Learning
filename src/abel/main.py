from transformers import pipeline

class AbelAnalyzer:
    def __init__(self, model_id):
        self.generator = pipeline("text-generation", model=model_id)

    def analyze_peer_data(self, metrics):
        prompt = f"The model loss is {metrics['loss']}. Is this good? "
        out = self.generator(prompt, max_new_tokens=50)
        return out[0]['generated_text']
    def analyze_peer_data(self, metrics):
        """
        Interprets metrics received from Caine.
        metrics: dict containing 'loss', 'epochs', 'trend'
        """
        # Contenxt Formula
        context = (
            f"Context: A peer model (Caine) just finished training a linear regression. "
            f"Final Loss: {metrics['loss']:.6f}. "
            f"Training Duration: {metrics['epochs']} epochs. "
            f"Trend: {metrics['trend']}.\n\n"
            f"Analysis Request: Based on these stats, what is your evaluation of the model's convergence?"
        )

        print("Abel: Processing peer data...")
        
        response = self.generator(
            context,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.2,
            truncation=True
        )

        return response[0]['generated_text'].replace(context, "").strip()

if __name__ == "__main__":
    # Simulated data received from Caine via TensorBoard
    caine_results = {
        "loss": 0.0042,
        "epochs": 50,
        "trend": "steadily decreasing"
    }

    abel = AbelAnalyzer()
    insight = abel.analyze_peer_data(caine_results)
    
    print("\n" + "="*30)
    print("ABEL'S EVALUATION:")
    print("="*30)
    print(insight)

    #begin to carry out task by reading soultion and implementing it using os controls to via control the android robot or system
    def carry_out_solution(solution_description):
        print(f"Carrying out solution: {solution_description}")
        
