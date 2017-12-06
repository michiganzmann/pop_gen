package scenario.gibbs;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.gibbs.problem.GibbsProblem;

public class GibbsAlgorithm {
	final private GibbsProblem problem;
	final private Random random;
	
	private int[] currentSample;
	private int currentDimension;
	
	public GibbsAlgorithm(GibbsProblem problem, Random random) {
		this.problem = problem;
		this.random = random;
		
		this.currentSample = problem.getInitialSample();
		this.currentDimension = 0;
	}
	
	private int select(INDArray probabilities) {
		INDArray cumulative = probabilities.cumsum(0);
		double r = random.nextDouble();
		
		int i = 0;
		while (r > cumulative.getDouble(i)) {
			i++;
		}
		
		return i;
	}
	
	public int[] next() {
		INDArray probabilities = problem.getProbabilities(currentDimension, currentSample);

		currentSample[currentDimension] = select(probabilities);
		
		currentDimension++;
		if (currentDimension == currentSample.length) {
			currentDimension = 0;
		}
		
		return currentSample;
	}
}
