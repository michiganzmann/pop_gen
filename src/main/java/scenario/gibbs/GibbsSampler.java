package scenario.gibbs;

import scenario.Sampler;

public class GibbsSampler implements Sampler {
	final private GibbsAlgorithm algorithm;
	final private int samplingRate;
	
	private int remainingBurnInSamples;
	
	public GibbsSampler(GibbsAlgorithm algorithm, int samplingRate, int burnInSamples) {
		this.algorithm = algorithm;
		this.samplingRate = samplingRate;
		this.remainingBurnInSamples = burnInSamples;
	}
	
	public int[] sample() {
		while (remainingBurnInSamples > 0) {
			algorithm.next();
			remainingBurnInSamples--;
		}
		
		for (int i = 0; i < samplingRate - 1; i++) {
			algorithm.next();
		}
		
		return algorithm.next();		
	}
}
