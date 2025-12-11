"""
Ecosystem Restoration Scenario Optimizer
Multi-objective optimization using NSGA-II to find optimal restoration strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'features'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class EcosystemRestorationProblem(Problem):
    """
    Multi-objective optimization problem for ecosystem restoration
    
    Decision Variables (interventions):
    - x1: Green cover increase (%) [0-20]
    - x2: Tree plantation density (trees/km²) [0-10000]
    - x3: Vehicle emission reduction (%) [0-50]
    - x4: Industrial emission control (%) [0-40]
    - x5: Water quality improvement (%) [0-30]
    - x6: Biodiversity enhancement budget (million ₹) [0-100]
    
    Objectives (minimize):
    - f1: PM2.5 concentration
    - f2: Total implementation cost
    - f3: Time to achieve target (years)
    
    Constraints:
    - Total budget < 1000 million ₹
    """
    
    def __init__(self, baseline_data):
        self.baseline = baseline_data
        
        # Cost coefficients (million ₹)
        self.costs = {
            'green_cover': 50,      # per % increase
            'tree_plantation': 0.005,  # per tree
            'vehicle_control': 20,   # per % reduction
            'industrial_control': 30, # per % reduction
            'water_improvement': 15,  # per % improvement
            'biodiversity': 1        # direct budget
        }
        
        # Effectiveness coefficients (PM2.5 reduction per unit intervention)
        self.effectiveness = {
            'green_cover': 0.8,      # PM2.5 reduction per % green cover
            'tree_plantation': 0.0003,  # PM2.5 reduction per tree/km²
            'vehicle_control': 0.6,   # PM2.5 reduction per % emission reduction
            'industrial_control': 0.5, # PM2.5 reduction per % emission control
            'water_improvement': 0.2,  # Indirect effect
            'biodiversity': 0.1       # Indirect effect per million ₹
        }
        
        # Implementation time (years per unit intervention)
        self.time_coefficients = {
            'green_cover': 0.3,
            'tree_plantation': 0.0002,
            'vehicle_control': 0.1,
            'industrial_control': 0.15,
            'water_improvement': 0.2,
            'biodiversity': 0.05
        }
        
        super().__init__(
            n_var=6,
            n_obj=3,
            n_constr=1,
            xl=np.array([0, 0, 0, 0, 0, 0]),
            xu=np.array([20, 10000, 50, 40, 30, 100])
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate objectives and constraints"""
        
        # Extract decision variables
        green_cover = X[:, 0]
        tree_density = X[:, 1]
        vehicle_reduction = X[:, 2]
        industrial_control = X[:, 3]
        water_improvement = X[:, 4]
        biodiversity_budget = X[:, 5]
        
        # Objective 1: PM2.5 concentration (minimize)
        baseline_pm25 = self.baseline['PM2.5'].mean()
        
        pm25_reduction = (
            green_cover * self.effectiveness['green_cover'] +
            tree_density * self.effectiveness['tree_plantation'] +
            vehicle_reduction * self.effectiveness['vehicle_control'] +
            industrial_control * self.effectiveness['industrial_control'] +
            water_improvement * self.effectiveness['water_improvement'] +
            biodiversity_budget * self.effectiveness['biodiversity']
        )
        
        # Add synergy effects (non-linear benefits)
        synergy = 0.1 * np.sqrt(green_cover * tree_density / 1000)
        pm25_reduction += synergy
        
        f1 = baseline_pm25 - pm25_reduction
        f1 = np.maximum(f1, 10)  # Minimum achievable PM2.5
        
        # Objective 2: Total implementation cost (minimize, million ₹)
        f2 = (
            green_cover * self.costs['green_cover'] +
            tree_density * self.costs['tree_plantation'] +
            vehicle_reduction * self.costs['vehicle_control'] +
            industrial_control * self.costs['industrial_control'] +
            water_improvement * self.costs['water_improvement'] +
            biodiversity_budget * self.costs['biodiversity']
        )
        
        # Objective 3: Implementation time (minimize, years)
        f3 = (
            green_cover * self.time_coefficients['green_cover'] +
            tree_density * self.time_coefficients['tree_plantation'] +
            vehicle_reduction * self.time_coefficients['vehicle_control'] +
            industrial_control * self.time_coefficients['industrial_control'] +
            water_improvement * self.time_coefficients['water_improvement'] +
            biodiversity_budget * self.time_coefficients['biodiversity']
        )
        f3 = np.maximum(f3, 1)  # Minimum 1 year
        
        # Constraint: Total budget < 1000 million ₹
        g1 = f2 - 1000
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1])


class RestorationOptimizer:
    """Optimize ecosystem restoration strategies"""
    
    def __init__(self):
        self.baseline_data = None
        self.problem = None
        self.result = None
        
    def load_baseline_data(self):
        """Load current ecosystem state"""
        logger.info("\n" + "=" * 70)
        logger.info("LOADING BASELINE DATA")
        logger.info("=" * 70)
        
        master_path = DATA_DIR / 'master_dataset.parquet'
        self.baseline_data = pd.read_parquet(master_path)
        
        # Use latest year data as baseline
        latest_year = self.baseline_data['date'].max().year
        self.baseline_data = self.baseline_data[
            self.baseline_data['date'].dt.year == latest_year
        ]
        
        logger.info(f"✓ Loaded baseline data from {latest_year}")
        logger.info(f"  Records: {len(self.baseline_data)}")
        logger.info(f"  Avg PM2.5: {self.baseline_data['PM2.5'].mean():.2f} µg/m³")
        logger.info(f"  Avg AQI: {self.baseline_data['AQI'].mean():.1f}")
        logger.info(f"  Ecosystem Health Score: {self.baseline_data['Ecosystem_Health_Score'].mean():.1f}")
        
    def optimize(self, population_size=100, n_generations=100):
        """Run multi-objective optimization"""
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING MULTI-OBJECTIVE OPTIMIZATION")
        logger.info("=" * 70)
        
        # Define problem
        self.problem = EcosystemRestorationProblem(self.baseline_data)
        
        # Configure NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Termination criterion
        termination = get_termination("n_gen", n_generations)
        
        logger.info(f"Algorithm: NSGA-II")
        logger.info(f"  Population size: {population_size}")
        logger.info(f"  Generations: {n_generations}")
        logger.info(f"  Objectives: 3 (PM2.5, Cost, Time)")
        logger.info(f"  Decision variables: 6")
        logger.info("\nOptimizing...")
        
        # Run optimization
        self.result = minimize(
            self.problem,
            algorithm,
            termination,
            seed=42,
            verbose=False
        )
        
        logger.info(f"\n✓ Optimization complete!")
        logger.info(f"  Solutions in Pareto front: {len(self.result.F)}")
        
    def analyze_results(self):
        """Analyze and display optimization results"""
        logger.info("\n" + "=" * 70)
        logger.info("ANALYZING PARETO-OPTIMAL SOLUTIONS")
        logger.info("=" * 70)
        
        # Extract Pareto-optimal solutions
        pareto_decisions = self.result.X
        pareto_objectives = self.result.F
        
        # Create results DataFrame
        results_df = pd.DataFrame(
            pareto_decisions,
            columns=[
                'Green_Cover_Increase_%',
                'Tree_Density_per_km2',
                'Vehicle_Emission_Reduction_%',
                'Industrial_Control_%',
                'Water_Quality_Improvement_%',
                'Biodiversity_Budget_Million_Rs'
            ]
        )
        
        results_df['PM2.5_Target'] = pareto_objectives[:, 0]
        results_df['Total_Cost_Million_Rs'] = pareto_objectives[:, 1]
        results_df['Implementation_Time_Years'] = pareto_objectives[:, 2]
        
        # Identify interesting scenarios
        scenarios = {
            'Fastest': results_df.loc[results_df['Implementation_Time_Years'].idxmin()],
            'Cheapest': results_df.loc[results_df['Total_Cost_Million_Rs'].idxmin()],
            'Best_Air_Quality': results_df.loc[results_df['PM2.5_Target'].idxmin()],
            'Balanced': results_df.loc[
                ((results_df - results_df.min()) / (results_df.max() - results_df.min()))
                [['PM2.5_Target', 'Total_Cost_Million_Rs', 'Implementation_Time_Years']].sum(axis=1).idxmin()
            ]
        }
        
        # Display scenarios
        logger.info("\nKEY RESTORATION SCENARIOS:")
        logger.info("=" * 70)
        
        for name, scenario in scenarios.items():
            logger.info(f"\n{name.upper()} SCENARIO:")
            logger.info(f"  Interventions:")
            logger.info(f"    • Green cover increase: {scenario['Green_Cover_Increase_%']:.1f}%")
            logger.info(f"    • Tree plantation: {scenario['Tree_Density_per_km2']:.0f} trees/km²")
            logger.info(f"    • Vehicle emission reduction: {scenario['Vehicle_Emission_Reduction_%']:.1f}%")
            logger.info(f"    • Industrial control: {scenario['Industrial_Control_%']:.1f}%")
            logger.info(f"    • Water quality improvement: {scenario['Water_Quality_Improvement_%']:.1f}%")
            logger.info(f"    • Biodiversity budget: ₹{scenario['Biodiversity_Budget_Million_Rs']:.1f}M")
            logger.info(f"  Expected outcomes:")
            logger.info(f"    • PM2.5 target: {scenario['PM2.5_Target']:.1f} µg/m³")
            logger.info(f"    • Total cost: ₹{scenario['Total_Cost_Million_Rs']:.1f}M")
            logger.info(f"    • Implementation time: {scenario['Implementation_Time_Years']:.1f} years")
        
        # Save results
        results_path = RESULTS_DIR / 'restoration_scenarios.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n✓ All {len(results_df)} Pareto-optimal scenarios saved to: {results_path}")
        
        # Save key scenarios
        scenarios_df = pd.DataFrame(scenarios).T
        scenarios_path = RESULTS_DIR / 'key_scenarios.csv'
        scenarios_df.to_csv(scenarios_path)
        logger.info(f"✓ Key scenarios saved to: {scenarios_path}")
        
        return results_df, scenarios
        
    def visualize_results(self, results_df):
        """Create visualizations of Pareto front"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        # 1. 3D Pareto front
        fig = plt.figure(figsize=(15, 10))
        
        ax = fig.add_subplot(221, projection='3d')
        scatter = ax.scatter(
            results_df['PM2.5_Target'],
            results_df['Total_Cost_Million_Rs'],
            results_df['Implementation_Time_Years'],
            c=results_df['PM2.5_Target'],
            cmap='RdYlGn_r',
            s=100,
            alpha=0.6
        )
        ax.set_xlabel('PM2.5 Target (µg/m³)', fontsize=10)
        ax.set_ylabel('Total Cost (₹M)', fontsize=10)
        ax.set_zlabel('Time (years)', fontsize=10)
        ax.set_title('3D Pareto Front', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='PM2.5', shrink=0.5)
        
        # 2. Cost vs PM2.5
        ax2 = fig.add_subplot(222)
        ax2.scatter(
            results_df['Total_Cost_Million_Rs'],
            results_df['PM2.5_Target'],
            c=results_df['Implementation_Time_Years'],
            cmap='viridis',
            s=100,
            alpha=0.6
        )
        ax2.set_xlabel('Total Cost (₹M)', fontsize=10)
        ax2.set_ylabel('PM2.5 Target (µg/m³)', fontsize=10)
        ax2.set_title('Cost vs Air Quality Trade-off', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Time vs PM2.5
        ax3 = fig.add_subplot(223)
        ax3.scatter(
            results_df['Implementation_Time_Years'],
            results_df['PM2.5_Target'],
            c=results_df['Total_Cost_Million_Rs'],
            cmap='plasma',
            s=100,
            alpha=0.6
        )
        ax3.set_xlabel('Implementation Time (years)', fontsize=10)
        ax3.set_ylabel('PM2.5 Target (µg/m³)', fontsize=10)
        ax3.set_title('Time vs Air Quality Trade-off', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Intervention mix heatmap
        ax4 = fig.add_subplot(224)
        intervention_cols = [
            'Green_Cover_Increase_%',
            'Tree_Density_per_km2',
            'Vehicle_Emission_Reduction_%',
            'Industrial_Control_%',
            'Water_Quality_Improvement_%',
            'Biodiversity_Budget_Million_Rs'
        ]
        
        # Normalize for visualization
        normalized = results_df[intervention_cols].copy()
        for col in normalized.columns:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
        
        # Take top 20 solutions by PM2.5
        top_solutions = normalized.iloc[results_df['PM2.5_Target'].nsmallest(20).index]
        
        sns.heatmap(
            top_solutions.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Normalized Intensity'},
            ax=ax4,
            yticklabels=[col.replace('_', ' ') for col in intervention_cols]
        )
        ax4.set_xlabel('Solution Index', fontsize=10)
        ax4.set_title('Intervention Mix (Top 20 Solutions)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        viz_path = RESULTS_DIR / 'pareto_front_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Pareto front visualization saved to: {viz_path}")
        plt.close()
        
        # 5. Intervention importance
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(intervention_cols):
            ax = axes[idx]
            ax.scatter(
                results_df[col],
                results_df['PM2.5_Target'],
                c=results_df['Total_Cost_Million_Rs'],
                cmap='coolwarm',
                alpha=0.6
            )
            ax.set_xlabel(col.replace('_', ' '), fontsize=9)
            ax.set_ylabel('PM2.5 Target (µg/m³)', fontsize=9)
            ax.set_title(f'{col.split("_")[0]} Impact', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        impact_path = RESULTS_DIR / 'intervention_impact.png'
        plt.savefig(impact_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Intervention impact analysis saved to: {impact_path}")
        plt.close()


def main():
    """Main optimization workflow"""
    logger.info("=" * 70)
    logger.info("ECOSYSTEM RESTORATION OPTIMIZATION SYSTEM")
    logger.info("=" * 70)
    
    optimizer = RestorationOptimizer()
    
    # Load data
    optimizer.load_baseline_data()
    
    # Run optimization
    optimizer.optimize(population_size=100, n_generations=100)
    
    # Analyze results
    results_df, scenarios = optimizer.analyze_results()
    
    # Visualize
    optimizer.visualize_results(results_df)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESTORATION OPTIMIZATION COMPLETE ✓")
    logger.info("=" * 70)
    logger.info(f"\nResults saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
