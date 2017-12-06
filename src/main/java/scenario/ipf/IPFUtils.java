package scenario.ipf;

import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class IPFUtils {
	static public INDArrayIndex[] getIndices(int numberOfDimensions, int[] dimensions, int categories[]) {
		INDArrayIndex[] index = new INDArrayIndex[numberOfDimensions];
		
		for (int i = 0; i < numberOfDimensions; i++) {
			index[i] = NDArrayIndex.all();;
		}
		
		for (int i = 0; i < dimensions.length; i++) {
			index[dimensions[i]] = NDArrayIndex.point(categories[i]);
		}
		
		return index;
	}
}
