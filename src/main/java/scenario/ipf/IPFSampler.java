package scenario.ipf;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;

import scenario.Sampler;

public class IPFSampler implements Sampler {
	final private INDArray weights;
	final private Random random;
	
	public IPFSampler(INDArray weights, Random random) {
		this.weights = weights.div(weights.sumNumber());
		this.random = random;
	}
	
	public int[] sample() {
		int[] index = new int[weights.shape().length];
		
		while (true) {
			for (int i = 0; i < weights.shape().length; i++) {
				index[i] = random.nextInt(weights.shape()[i]);
			}
			
			if (random.nextDouble() <= weights.getDouble(index)) {
				return index;
			}
		}
	}
}
