package scenario.gibbs.problem;

import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class GibbsProblemFromCounts implements GibbsProblem {
	final private INDArray counts;
	private int[] initialSample;
	
	public GibbsProblemFromCounts(INDArray counts) {
		this.counts = counts;
		initialSample = new int[counts.shape().length];
	}
	
	@Override
	public int[] getInitialSample() {
		return initialSample;
	}
	
	public void chooseRandomInitialSample(Random random) {
		while (true) {
			int index[] = new int[counts.shape().length];
			
			for (int i = 0; i < index.length; i++) {
				index[i] = random.nextInt(counts.shape()[i]);
			}
			
			if (counts.getDouble(index) > 0.0) {
				initialSample = index;
				return;
			}
		}
	}
	
	@Override
	public INDArray getProbabilities(int dimension, int[] conditionals) {
		INDArrayIndex[] indices = new INDArrayIndex[counts.shape().length];

		for (int i = 0; i < indices.length; i++) {
			indices[i] = i == dimension ? NDArrayIndex.all() : NDArrayIndex.point(conditionals[i]);
		}
		
		INDArray slice = counts.get(indices);
		slice = slice.div(slice.sumNumber());
		
		return slice;
	}
}
