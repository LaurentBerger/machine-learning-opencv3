#include <iostream>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;

using namespace std;

bool plotSupportVectors=false;
int numTrainingPoints=200;
int numTestPoints=2000;
int sizes=200;
int eq=0;

// accuracy
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
	assert(predicted.rows == actual.rows);
    if (predicted.type() == CV_64F && actual.type() == CV_32F)
    {
        int t = 0;
        int f = 0;
        for (int i = 0; i < actual.rows; i++) {
            float p = predicted.at<double>(i, 0);
            float a = actual.at<float>(i, 0);
            if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
                t++;
            }
            else {
                f++;
            }
        }
        return (t * 1.0) / (t + f);
    }
    if (predicted.type() == CV_32F && actual.type() == CV_32F)
    {
        int t = 0;
        int f = 0;
        for (int i = 0; i < actual.rows; i++) {
            float p = predicted.at<float>(i, 0);
            float a = actual.at<float>(i, 0);
            if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
                t++;
            }
            else {
                f++;
            }
        }
        return (t * 1.0) / (t + f);
    }
    if (predicted.type() == CV_32S && actual.type() == CV_32S)
    {
	    int t = 0;
	    int f = 0;
	    for(int i = 0; i < actual.rows; i++) {
		    int p = predicted.at<int>(i,0);
		    int a = actual.at<int>(i,0);
		    if((p == a )) {
			    t++;
		    } else {
			    f++;
		    }
	    }
	    return (t * 1.0) / (t + f);
    }
    if (predicted.type() == CV_32S && actual.type() == CV_32F)
    {
	    int t = 0;
	    int f = 0;
	    for(int i = 0; i < actual.rows; i++) {
		    int p = predicted.at<int>(i,0);
		    float a = actual.at<float>(i,0);
		    if((p == static_cast<int>(a) )) {
			    t++;
		    } else {
			    f++;
		    }
	    }
	    return (t * 1.0) / (t + f);
    }
}

// plot data and class
void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	cv::Mat plot(sizes, sizes, CV_8UC3);
	plot.setTo(cv::Scalar(255.0,255.0,255.0));
    if (classes.type()==CV_32F)
	    for(int i = 0; i < data.rows; i++) 
        {

		    float x = data.at<float>(i,0) * sizes;
		    float y = data.at<float>(i,1) * sizes;

		    if(classes.at<float>(i, 0) > 0) 
            {
			    cv::circle(plot, Point(x,y), 2, CV_RGB(255,0,0),1);
		    } else 
            {
			    cv::circle(plot, Point(x,y), 2, CV_RGB(0,255,0),1);
		    }
	    }
    else if (classes.type() == CV_64F)
        for (int i = 0; i < data.rows; i++) 
        {

            float x = data.at<float>(i, 0) * sizes;
            float y = data.at<float>(i, 1) * sizes;
            double a= classes.at<double>(i, 0),b= classes.at<double>(i, 1),c= classes.at<double>(i, 2);
            if (classes.at<double>(i, 0) > 0) 
            {
                cv::circle(plot, Point(x, y), 2, CV_RGB(255, 0, 0), 1);
            }
            else 
            {
                cv::circle(plot, Point(x, y), 2, CV_RGB(0, 255, 0), 1);
            }
        }
    namedWindow(name,WINDOW_NORMAL);
	cv::imshow(name, plot);
}

// function to learn
int f(float x, float y, int equation) {
	switch(equation) {
	case 0:
		return y > sin(x*10) ? -1 : 1;
		break;
	case 1:
		return y > cos(x * 10) ? -1 : 1;
		break;
	case 2:
		return y > 2*x ? -1 : 1;
		break;
	case 3:
		return y > tan(x*10) ? -1 : 1;
		break;
	default:
		return y > cos(x*10) ? -1 : 1;
	}
}

// label data with equation
cv::Mat labelData(cv::Mat points, int equation) {
	cv::Mat labels(points.rows, 1, CV_32FC1);
	for(int i = 0; i < points.rows; i++) {
			 float x = points.at<float>(i,0);
			 float y = points.at<float>(i,1);
			 labels.at<float>(i, 0) = f(x, y, equation);
		}
	return labels;
}

void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
    Ptr<SVM> svm=SVM::create();
    Mat tc;
    trainingClasses.convertTo(tc,CV_32S);
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY);//CvSVM::RBF, CvSVM::LINEAR ...
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER +CV_TERMCRIT_EPS, 1000, 1e-6));
	svm->setDegree(0.5); // for poly
	svm->setGamma(1); // for poly/rbf/sigmoid
	svm->setCoef0(0); // for poly/sigmoid

	svm->setC(7); // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	svm->setNu(0.5); // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	svm->setP(0.0); // for CV_SVM_EPS_SVR

	//svm->setClassWeights(Mat()); // for CV_SVM_C_SVC
    Ptr<cv::ml::TrainData> t = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE,tc);
	// SVM training (use train auto for OpenCV>=2.0)
//	svm->train(trainingData,SampleTypes::ROW_SAMPLE,tc);
	svm->trainAuto(t);

	cv::Mat predicted(testClasses.rows, 1, CV_32F);
    svm->predict(testData,predicted);
/*	for(int i = 0; i < testData.rows; i++) {
		cv::Mat sample = testData.row(i);

		float x = sample.at<float>(0,0);
		float y = sample.at<float>(0,1);

		predicted.at<float>(i, 0) = svm->predict(sample);
	}*/

	cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions SVM");

	// plot support vectors
	if(plotSupportVectors) 
    {
		cv::Mat plot_sv(sizes, sizes, CV_8UC3);
		plot_sv.setTo(cv::Scalar(255.0,255.0,255.0));

        Mat vec = svm->getUncompressedSupportVectors();
		int svec_count = vec.rows;
		for(int vecNum = 0; vecNum < svec_count; vecNum++) 
        {
//			vec = svm.get_support_vector(vecNum);
//			cv::circle(plot_sv, Point(vec[0]*size, vec[1]*size), 3 , CV_RGB(0, 0, 0));
		}
	cv::imshow("Support Vectors", plot_sv);
	}
}

void mlp(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	cv::Mat layers = cv::Mat(4, 1, CV_32SC1);

	layers.at<int>(0,0) = 2;
	layers.at<int>(1,0) = 10;
	layers.at<int>(2,0) = 20;
	layers.at<int>(3,0) = 1;

	Ptr<ANN_MLP> mlp=ANN_MLP::create();;
    mlp->setLayerSizes(layers);
    mlp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
    mlp->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
    mlp->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER +CV_TERMCRIT_EPS, 1000, 1e-6));
    mlp->setBackpropWeightScale(0.05f);
    mlp->setBackpropMomentumScale(0.05f);

	// train
    Ptr<cv::ml::TrainData> t = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE,trainingClasses);
	mlp->train(t);

	cv::Mat predicted;//(testClasses.rows, 1, CV_32SC1);
    mlp->predict(testData, predicted);

	cout << "Accuracy_{MLP} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions Backpropagation");
}

void knn(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses, int K) {

    Ptr<KNearest> knn = KNearest::create();

    knn->train(trainingData, ROW_SAMPLE, trainingClasses);

    cv::Mat predicted;;
    knn->findNearest(testData, K, predicted);
    knn->predict(testData,  predicted);

    cout << "Accuracy_{KNN} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions KNN");

}

void em(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses,int K) {

    Ptr<EM> emModel = EM::create();
    Mat labels;
    emModel->setClustersNumber(K);
    emModel->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
    emModel->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 300, 0.1));
    emModel->trainEM(trainingData, noArray(), labels, noArray());

    cv::Mat predict,predicted(testData.rows,1,CV_32FC1);
    emModel->predict(testData,  predict);
    for (int i = 0; i < predict.rows; i++)
    {
        int j[2];
        double maxVal;
        minMaxIdx(predict.row(i),NULL,&maxVal,NULL,j);
        predicted.at<float>(i,0)=j[1];
    }
    

    cout << "Accuracy_{EM} = " << evaluate(predicted, testClasses) << endl;
    plot_binary(testData, predicted, "Predictions EM");

}

void bayes(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	Ptr<NormalBayesClassifier> bayes=NormalBayesClassifier::create();
    Mat tc;
    trainingClasses.convertTo(tc,CV_32S);
    Ptr<cv::ml::TrainData> t = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE,tc);
    bayes->train(t);
	cv::Mat predicted;;
	bayes->predict(testData,predicted);
	
	cout << "Accuracy_{BAYES} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions Bayes");

}


void decisiontree(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {

	Ptr<DTrees> dtree=DTrees::create();
    
    dtree->setMaxDepth(8);
    dtree->setMinSampleCount(2);
    dtree->setUseSurrogates(false);
    dtree->setCVFolds(0);
    dtree->setUse1SERule(true);
    dtree->setTruncatePrunedTree(false);
	cv::Mat var_type(1, 3, CV_8U);
    
    Ptr<cv::ml::TrainData> t = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE,trainingClasses);

	dtree->train(t);
	cv::Mat predicted;
    dtree->predict(testData,predicted);

	cout << "Accuracy_{TREE} = " << evaluate(predicted, testClasses) << endl;
	plot_binary(testData, predicted, "Predictions tree");

}
void callbackButton(int state, void *userdata)
{
}
void callbackButton1(int state, void *userdata)
{
}

void callbackButton2(int state, void *userdata)
{
}

void on_mouse (int event, int x, int y, int flags, void *userdata)
{

}
int main() {

    
    
    cv::ocl::setUseOpenCL(false);
	cv::Mat trainingData(numTrainingPoints, 2, CV_32FC1);
	cv::Mat testData(numTestPoints, 2, CV_32FC1);

	cv::randu(trainingData,0,1);
	cv::randu(testData,0,1);

	cv::Mat trainingClasses = labelData(trainingData, eq);
	cv::Mat testClasses = labelData(testData, eq);

	plot_binary(trainingData, trainingClasses, "Training Data");
	plot_binary(testData, testClasses, "Test Data");

    em(trainingData, trainingClasses, testData, testClasses,2);
	bayes(trainingData, trainingClasses, testData, testClasses);
	svm(trainingData, trainingClasses, testData, testClasses);
	mlp(trainingData, trainingClasses, testData, testClasses);
	knn(trainingData, trainingClasses, testData, testClasses, 2);
    decisiontree(trainingData, trainingClasses, testData, testClasses);

	cv::waitKey();

	return 0;
}


