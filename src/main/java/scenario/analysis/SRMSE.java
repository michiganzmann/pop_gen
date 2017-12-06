package scenario.analysis;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SRMSE {
	final INDArray reference;
	final INDArray sampleCounts;
	
	public SRMSE(INDArray reference) {
		this.reference = reference;
		this.sampleCounts = Nd4j.zeros(reference.shape());
	}
	
	public void addSample(int[] sample) {
		INDArrayIndex[] index = new INDArrayIndex[sample.length];
		
		for (int i = 0; i < sample.length; i++) {
			index[i] = NDArrayIndex.point(sample[i]);
		}
		
		sampleCounts.get(index).addi(1.0);
	}
	
	public double compute() {
		INDArray referenceFrequencies = reference.div(reference.sumNumber());
		INDArray sampleFrequencies = sampleCounts.div(sampleCounts.sumNumber());
		
		INDArray difference = referenceFrequencies.sub(sampleFrequencies);
		difference = difference.mul(difference);
		
		double value = Math.sqrt(difference.sumNumber().doubleValue());
		
		for (int i = 0; i < reference.shape().length; i++) {
			value *= Math.sqrt(reference.shape()[i]);
		}
		
		return value;
	}
}
