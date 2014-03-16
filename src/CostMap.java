import com.googlecode.javacv.cpp.opencv_core;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class CostMap
{
    public static class Node{
        public short row;
        public short col;
        public short cost;
    }
    public opencv_core.CvMat image;

    public CostMap(opencv_core.CvMat image){
        this.image = image;
    }
}
