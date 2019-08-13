//package zfmllib.lsh;
//
//import netflix.memreader.*;
//import netflix.rmse.*;
//import java.io.*;
//import java.util.*;
//import cern.colt.map.OpenIntIntHashMap;
//import cern.colt.list.*;
//
//
////=============================================================================
//// This code is a (nearly) direct port from a C program by Timely Development
//// (www.timelydevelopment.com). The following notices and attributes were
//// distributed with the original source, and should be retained after
//// any modifications.
////
//// @author sowellb
////
//// Copyright (C) 2007 Timely Development (www.timelydevelopment.com)
////
//// Special thanks to Simon Funk and others from the Netflix Prize contest
//// for providing pseudo-code and tuning hints.
////
//// Feel free to use this code as you wish as long as you include
//// these notices and attribution.
////
//// Also, if you have alternative types of algorithms for accomplishing
//// the same goal and would like to contribute, please share them as well :)
////
//// STANDARD DISCLAIMER:
////
//// - THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY
//// - OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT
//// - LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
//// - FITNESS FOR A PARTICULAR PURPOSE.
////
////=============================================================================
//public class IncrementalSVDTest implements Serializable {
//
//    private static final long serialVersionUID = 1526472295622776147L;
//
//
//    //Global counts. The NUM_RATING count is for ua/ubbase
//    private final int NUM_RATINGS = 34; //that is, the total number of items in the matrix
//
//    static private final int NUM_USERS =7;
//    // row number in the matrix; //one extra number should be added, .e.g, 1001 for 1000 users
//
//    private final int NUM_MOVIES = 30;
//    // column number in the matrix;//one extra number should be added, .e.g, 1001 for 1000 movies
//
//    static private final int NUM_FEATURES = 3; //the number of features in the reduced matrix
//
//    private final int MIN_EPOCHS = 120; // the iteration times before getting the minimal improvement
////  private final int MAX_EPOCHS = 200;
//
//    private final float INIT_VALUE = 0.1f;
//    private final float MIN_IMPROVEMENT = 0.0001f;
//    private final float LRATE = 0.001f; //learning rate, the best learning rate the this dataset
//    private final float K = 0.015f; //a regularization coefficient k
//
//    static private int[] uid;
//    private short[] mid;
//    private byte[] rating;
//    private float[] cache;
//
//
//    static private float[][] movieFeatures;
//    static private float[][] userFeatures;
//
//
//
//    /**
//     * Default constructor. Initializes
//     * data structures.
//     */
//    public IncrementalSVDTest() {
//
//        uid = new int[NUM_RATINGS]; // not the actual id, just for ratings
//        mid = new short[NUM_RATINGS];
//        rating = new byte[NUM_RATINGS];
//        cache = new float[NUM_RATINGS];
//
//        movieFeatures = new float[NUM_FEATURES][NUM_MOVIES];
//        userFeatures = new float[NUM_FEATURES][NUM_USERS]; //actuall store the features,but no user ids
//
//
//        for(int i = 0; i < NUM_FEATURES; i++) {
//            for(int j = 0; j < NUM_MOVIES; j++) {
//                movieFeatures[i][j] = INIT_VALUE;
//            }
//            for(int j = 0; j < NUM_USERS; j++) {
//                userFeatures[i][j] = INIT_VALUE;
//            }
//        }
//    }
//
//
//    /**
//     * Train each feature on
//     * the entire data set.
//     */
//    private void calcFeatures() {
//
////        Rating rating;
//        double err, p, sq, rmse_last = 2.0, rmse = 2.0;
//        short currMid;
//        int currUid;
//        float cf, mf;
//
//        for(int i = 0; i < NUM_FEATURES; i++)
//        {
//
//
//            for(int j = 0; (j < MIN_EPOCHS) || (rmse <= rmse_last - MIN_IMPROVEMENT); j++)
//            {
//
//                sq = 0;
//                rmse_last = rmse;
//
//                for(int k = 0; k < rating.length; k++)
//                {
//
//                    currMid = mid[k];
//                    currUid = uid[k];
//
//                    // Predict rating and calculate error
//                    p = predictRating(currMid, currUid, i, cache[k], true);
//                    err = (1.0 * rating[k] - p);
//                    sq += err*err;
//
//                    // Cache old features
//                    cf = userFeatures[i][currUid];
//                    mf = movieFeatures[i][currMid];
//
//                    // Cross-train
//                    userFeatures[i][currUid] += (float) (LRATE * (err * mf - K * cf));
//                    movieFeatures[i][currMid] += (float) (LRATE * (err * cf - K * mf));
//                }
//                rmse = Math.sqrt(sq/rating.length);
//            }
//
//            //Cache old predictions
//            for(int j=0; j<rating.length; j++) {
//                cache[j] = (float)predictRating(mid[j], uid[j], i, cache[j], false);
//            }
//
//            System.out.println("Calculated feature: " + (i+1));
//        }
//    }
//
//    /**
//     * For use during training.
//     *
//     * @param  mid  Movie id.
//     * @param  uid   User id.
//     * @param  feature  The feature we are training
//     * @param  cache  Cache value, if applicable
//     * @param  bTrailing
//     * @return The predicted rating for use during training.
//     */
//    private double predictRating(short mid, int uid, int feature,
//                                 float cache, boolean bTrailing) {
//        double sum;
//
//        if(cache > 0)
//            sum = cache;
//        else
//            sum = 1;
//
//        sum += movieFeatures[feature][mid] * userFeatures[feature][uid];
//
//        if(sum > 5)
//            sum = 5;
//        else if(sum < 1)
//            sum = 1;
//
//        if(bTrailing) {
//            sum += (NUM_FEATURES - feature - 1) * (INIT_VALUE*INIT_VALUE);
//
//            if(sum > 5)
//                sum = 5;
//            else if(sum < 1)
//                sum = 1;
//        }
//        return sum;
//    }
//
//    /**
//     * Predicts the rating for a user/movie pair using
//     * all features that have been trained.
//     *
//     * @param  mid  The movie to predict the rating for.
//     * @param  uid  The user to predict the rating for.
//     * @return The predicted rating for (uid, mid).
//     */
//    public  double predictRating(short mid, int uid) {
//
//        double sum = 1;
//
//        for(int i = 0; i < NUM_FEATURES; i++) {
//            sum += movieFeatures[i][mid] * userFeatures[i][uid];
//            if(sum > 5)
//                sum = 5;
//            else if(sum < 1)
//                sum = 1;
//        }
//
//        return sum;
//    }
//
//
//    /**
//     * Loads file containg all of the known
//     * ratings.
//     *
//     * @param  fileName  The data file.
//     */
//    public void loadData(String fileName) throws FileNotFoundException, IOException {
//
//        Scanner in = new Scanner(new File(fileName));
//
//        String[] line;
//        short currMid;
//        int currUid, newUid;
//        byte currRating;
//        String date;
//        int idCounter = 0, ratingCounter = 0;
//        while(in.hasNextLine()) {
//
//            line = in.nextLine().split(",");
//            currUid = Integer.parseInt(line[0]);
//            currMid = Short.parseShort(line[1]);
//            double r=Double.parseDouble(line[2]);
//            currRating = (byte) r;
//
//            mid[ratingCounter] = currMid;
//            uid[ratingCounter] = currUid;
//            rating[ratingCounter] = currRating;
//            ratingCounter++;
//        }
//    }
//
//
//
//
//
//    public static void Matrics_IncrementalSVDBuilder(String d, String t) {
//
//        try {
//            //Change to appropriate location.
//            String dataFile =d;
//            //"../datasets/training-100.dat";
//            String outputFile = t;
//            //"../datasets/test-100.dat";
//
//            IncrementalSVDTest incSVD = new IncrementalSVDTest();
//            incSVD.loadData(dataFile);
//            long startTime = System.currentTimeMillis();
//            incSVD.calcFeatures();
//            long endTime = System.currentTimeMillis();
//
//            //OK, remember the first line is never used, it is the initial value
//            BufferedWriter bw = new BufferedWriter(new FileWriter(new File(outputFile), true));
//
//            for(int i=1;i<NUM_USERS;i++)
//            {
//                for(int j=0;j<NUM_FEATURES;j++)
//                {
//                    //              System.out.print(userFeatures[j][i]+",");
//                    bw.write(userFeatures[j][i]+",");
//                }
//                //       System.out.println(i);
//                bw.write(String.valueOf(i));
//                bw.newLine();
//            }
//            bw.close();
//            System.out.println("Total time taken: " + (endTime - startTime));
//
//     /*           MemHelper mh = new MemHelper(testFile);
//                double rmse = incSVD.testWithMemHelper(mh);
//                System.out.println("RMSE = " + rmse); */
//            //          }
//        }
//        catch(FileNotFoundException e) {
//            System.out.println("Could not find file.");
//            System.out.println("usage: java IncrementalSVD serialFile");
//            e.printStackTrace();
//        }
//        catch(IOException e) {
//            System.out.println("Unknown IO error.");
//            e.printStackTrace();
//        }
//    }
//
//}
