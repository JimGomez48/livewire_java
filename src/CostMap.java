import com.googlecode.javacv.cpp.opencv_core.CvMat;

import java.util.ArrayList;

/**
 * WRITE CLASS DESCRIPTION HERE
 */
public class CostMap
{
    public static class Node{
        public short row;
        public short col;
        public int cost;
        public Node parent;

        public Node(){}

        public Node(short row, short col, byte cost, Node parent){
            this.row = row;
            this.col = col;
            this.cost = cost;
            this.parent = parent;
        }
    }

    private static final float RAD2 = 1.41421356f;
    Node[][] original;
    Node[][] costs;

    public CostMap(CvMat image){
        original = new Node[image.rows()][image.cols()];
        costs = new Node[image.rows()][image.cols()];
        reset(image);
    }

    public void reset(CvMat image){
        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                Node n = new Node();
                n.row = (short)i;
                n.col = (short)j;
                n.cost = (int)image.get(i, j);
                n.parent = null;
                original[i][j] = n;
            }
        }
    }

    public void reset(){
        for (int i = 0; i < original.length; i++) {
            for (int j = 0; j < original[0].length; j++) {
                costs[i][j] = original[i][j];
            }
        }
    }

    public void addSeed(int row, int col){
        //TODO
        expand(row, col);
    }

    private void expand(int row, int col){
        //TODO
    }

    private ArrayList<Node> getNeighbors(Node n){
        //TODO
        return null;
    }
    private int euclideanAdd(Node a, Node b){
        //TODO
        return 0;
    }

//    private int toIndex(int row, int col){
//        return original[0].length * row + col;
//    }
//
//    private int toIndex(Node n){
//        return original[0].length * n.row + n.col;
//    }
}
