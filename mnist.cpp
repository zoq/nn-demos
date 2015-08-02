#include <stdlib.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <mutex>
#include <boost/lexical_cast.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <thread>

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

#include <mlpack/methods/ann/connections/full_connection.hpp>
#include <mlpack/methods/ann/connections/identity_connection.hpp>
#include <mlpack/methods/ann/connections/bias_connection.hpp>
#include <mlpack/methods/ann/connections/conv_connection.hpp>
#include <mlpack/methods/ann/connections/pooling_connection.hpp>

#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/dropout_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/multiclass_classification_layer.hpp>

#include <mlpack/methods/ann/cnn.hpp>
#include <mlpack/methods/ann/trainer/trainer.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/optimizer/ada_delta.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "http.hpp"
#include "graphics.hpp"
#include "parser.hpp"
#include "config.hpp"

#include "fcgio.h"

using namespace mlpack;
using namespace mlpack::ann;


/*
 * Class to handle the training process of the specified neural network.
 */
class Job {
  public:
    /*
     * Create the job object to train the neural network on the specified
     * dataset.
     */
    Job(std::string content, arma::mat& X) : content(content), X(X)
    {
      nPoints = X.n_cols;
      // Build the target matrix.
      Y = arma::zeros<arma::mat>(10, nPoints);
      for (size_t i = 0; i < nPoints; i++)
      {
        if (i < nPoints / 2)
        {
          Y.col(i)(4) = 1;
        }
        else
        {
          Y.col(i)(9) = 1;
        }
      }

      // Create the network structure.
      inputLayerPtr = new NeuronLayer<RectifierFunction, arma::cube>(28, 28, 1);

      convLayer0Ptr = new ConvLayer<RectifierFunction>(24, 24, inputLayerPtr->LayerSlices(), 6);

      con1Ptr = new ConvConnection<NeuronLayer<RectifierFunction, arma::cube>,
                                               ConvLayer<RectifierFunction>,
                                               mlpack::ann::AdaDelta>
                                               (*inputLayerPtr, *convLayer0Ptr, 5);

      biasLayer0Ptr = new BiasLayer<>(6);

      con1BiasPtr = new BiasConnection<BiasLayer<>,
                                       ConvLayer<RectifierFunction>,
                                       mlpack::ann::AdaDelta,
                                       mlpack::ann::ZeroInitialization>
                                       (*biasLayer0Ptr, *convLayer0Ptr);

      poolingLayer0Ptr = new PoolingLayer<>(12, 12, inputLayerPtr->LayerSlices(), 6);

      con2Ptr = new PoolingConnection<ConvLayer<RectifierFunction>,
                                      PoolingLayer<> >
                                      (*convLayer0Ptr, *poolingLayer0Ptr);

      convLayer1Ptr = new ConvLayer<RectifierFunction>(8, 8, inputLayerPtr->LayerSlices(), 10);


      con3Ptr = new ConvConnection<PoolingLayer<>,
                                   ConvLayer<RectifierFunction>,
                                   mlpack::ann::AdaDelta>
                                   (*poolingLayer0Ptr, *convLayer1Ptr, 5);

      biasLayer3Ptr = new BiasLayer<>(10);

      con3BiasPtr = new BiasConnection<BiasLayer<>,
                                       ConvLayer<RectifierFunction>,
                                       mlpack::ann::AdaDelta,
                                       mlpack::ann::ZeroInitialization>
                                       (*biasLayer3Ptr, *convLayer1Ptr);

      poolingLayer1Ptr  = new PoolingLayer<>(4, 4, inputLayerPtr->LayerSlices(), 10);

      con4Ptr = new PoolingConnection<ConvLayer<RectifierFunction>,
                                      PoolingLayer<> >
                                      (*convLayer1Ptr, *poolingLayer1Ptr);

      outputLayerPtr = new SoftmaxLayer<arma::mat>(10, inputLayerPtr->LayerSlices());

      con5Ptr = new FullConnection<PoolingLayer<>,
                                   SoftmaxLayer<arma::mat>,
                                   mlpack::ann::AdaDelta>
                                   (*poolingLayer1Ptr, *outputLayerPtr);

      biasLayer1Ptr = new BiasLayer<>(1);

      con5BiasPtr = new FullConnection<BiasLayer<>,
                                       SoftmaxLayer<arma::mat>,
                                       mlpack::ann::AdaDelta,
                                       mlpack::ann::ZeroInitialization>
                                       (*biasLayer1Ptr, *outputLayerPtr);

      finalOutputLayerPtr = new MulticlassClassificationLayer();

      gradient0 = con1Ptr->Weights();
      gradient1 = con3Ptr->Weights();

      trainingError = 0;
      trainingErrorcurrent = 0;
      trainingSeqCur = 0;
      trainingSeqStart = 0;
      state = 1; // Waiting
    }

    /*
     * Run the job.
     */
    void Run()
    {
      state = 0; // Running

      auto module0 = std::tie(*con1Ptr, *con1BiasPtr);
      auto module1 = std::tie(*con2Ptr);
      auto module2 = std::tie(*con3Ptr, *con3BiasPtr);
      auto module3 = std::tie(*con4Ptr);
      auto module4 = std::tie(*con5Ptr, *con5BiasPtr);
      auto modules = std::tie(module0, module1, module2, module3, module4);

      CNN<decltype(modules), decltype(*finalOutputLayerPtr)>
      net(modules, *finalOutputLayerPtr);

      Trainer<decltype(net)> trainer(net, 1, 4, 0.03);

      arma::cube input = arma::cube(28, 28, nPoints);
      arma::cube inputPrediction = arma::cube(28, 28, 1);
      arma::mat predictionOutput;
      arma::Col<size_t> index = arma::linspace<arma::Col<size_t> >(0,
          nPoints - 1, nPoints);

      for (size_t i = 0; i < nPoints; i++)
        input.slice(i) = arma::mat(X.colptr(i), 28, 28);


      arma::mat error;
      int batchSize = 2;
      bool predict = true;
      for (size_t z = 0; z < 500; z++)
      {
        if (state == 2)
        {
          break;
        }

        if ((z % 10) == 0)
          trainingSeqStart++;

        trainingError = 0;
        predict = true;
        for (size_t i = 0; i < nPoints; i++)
        {
          index = arma::shuffle(index);

          arma::cube inputTemp = input.slices(index(i), index(i));
          arma::mat targetTemp = Y.col(index(i));
          net.FeedForward(inputTemp,  targetTemp, error);

          trainingError += net.Error();

          net.FeedBackward(error);

          if (!con1Ptr->Optimizer().Gradient().is_empty() && (con1Ptr->Optimizer().Gradient().max() > 0))
          {
            gradient0 = arma::cube(con1Ptr->Optimizer().Gradient());
          }

          if (!con3Ptr->Optimizer().Gradient().is_empty() && (con3Ptr->Optimizer().Gradient().max() > 0))
            gradient1 = arma::cube(con3Ptr->Optimizer().Gradient());

          if (((i + 1) % batchSize) == 0)
            net.ApplyGradients();

          if ((z % 4) == 0 && predict == true)
          {
            predict = false;
            for (size_t k = 0; k < 11; k++)
            {
              predictionInfo[k].image = new arma::mat(X.colptr(index(k)), 28, 28);
              inputPrediction.slice(0) = *predictionInfo[k].image;
              net.Predict(inputPrediction, predictionOutput);

              arma::uvec idx = arma::sort_index(predictionOutput.col(0), "descend");
              for (size_t j = 0; j < 5; j++)
              {
                predictionInfo[k].prediction[j] = std::to_string(idx(j));
                predictionInfo[k].prob[j] = std::to_string(predictionOutput(idx(j), 0));
              }
            }
          }
        }

        trainingError /= nPoints;

        trainingErrorcurrent = trainingError;
        sleep(1);
      }

      state = 2; // Stop
    }

    /*
     * The input matrix using the specified id to identify the layer.
     *
     * @param id The id of the layer.
     * @param m The matrix used to store the input activation.
     */
    void GetInputMatrix(int id, arma::mat& m)
    {
      if (id == 0)  // input image
      {
        m = inputLayerPtr->InputActivation();
      }
      else if (id >= 1 && id <= 6)
      {
        id -= 1;
        m = convLayer0Ptr->InputActivation().slice(id) * 255;
      }
      else if (id >= 7 && id <= 12)
      {
        id -= 7;
        m = poolingLayer0Ptr->InputActivation().slice(id) * 255;
      }
      else if(id >= 13 && id <= 22)
      {
        id -= 13;
        m = convLayer1Ptr->InputActivation().slice(id) * 255;
      }
      else if(id >= 23 && id <= 32)
      {
        id -= 23;
        m = poolingLayer1Ptr->InputActivation().slice(id) * 255;
      }
      else if(id >= 33 && id <= 42)
      {
        id -= 33;
        m = outputLayerPtr->InputActivation().row(id) * 255;
      }
      else
      {
        m = arma::mat(1, 1);
      }
    }

    /*
     * The weight matrix using the specified id to identify the layer.
     *
     * @param id The id of the layer.
     * @param m The matrix used to store the weight.
     */
    void GetWeightMatrix(int id, arma::mat& m)
    {
      if (id >= 0 && id <= 5)
      {
        if (con1Ptr->Weights().is_empty())
        {
          m = arma::mat(1, 1);
        }
        else
        {
          m = arma::normalise(con1Ptr->Weights().slice(id)) * 255;
        }
      }
      else if (id >= 6 && id <= 65)
      {
        id -= 6;
        if (con3Ptr->Weights().is_empty())
        {
          m = arma::mat(1, 1);
        }
        else
        {
          m = arma::normalise(con3Ptr->Weights().slice(id)) * 255;
        }
      }
      else
      {
        m = arma::mat(1, 1);
      }
    }

    /*
     * The gradient matrix using the specified id to identify the layer.
     *
     * @param id The id of the layer.
     * @param m The matrix used to store the gradient.
     */
    void GetGradientMatrix(int id, arma::mat& m)
    {
      if (id >= 0 && id <= 5)
      {
        if (gradient0.is_empty())
        {
          m = arma::mat(1, 1);
        }
        else
        {
          m = arma::normalise(gradient0.slice(id)) * 255;
        }
      }
      else if (id >= 6 && id <= 65)
      {
        id -= 6;
        if (gradient1.is_empty())
        {
          m = arma::mat(1, 1);
        }
        else
        {
          m = arma::normalise(gradient1.slice(id)) * 255;
        }
      }
      else
      {
        m = arma::mat(1, 1);
      }
    }

    /*
     * Info about the training process.
     *
     * @param id The id of the layer.
     */
    void GetInfoText(int id)
    {
      // std::cout.setf(std::ios::fixed, std::ios::floatfield);
      // std::cout.setf(std::ios::showpoint);

      std::cout << "{"
                << "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

      if (trainingSeqCur != trainingSeqStart) {
        std::cout << "\"trainingError\": \"" << std::fixed << std::setprecision(2)
                  << trainingErrorcurrent << "\",";
        std::cout.unsetf(std::ios_base::fixed);
      }



      if (!con1Ptr->Weights().is_empty())
      {
        std::cout << "\"maxWeightLayer1\": \"" << con1Ptr->Weights().max() << "\","
                  << "\"minWeightLayer1\": \"" << con1Ptr->Weights().min() << "\",";
      }

      if (!con3Ptr->Weights().is_empty())
      {
        std::cout << "\"maxWeightLayer3\": \"" << con3Ptr->Weights().max() << "\","
                  << "\"minWeightLayer3\": \"" << con3Ptr->Weights().min() << "\",";
      }

      if (!gradient0.is_empty())
      {
        std::cout << "\"maxGradientLayer1\": \"" << gradient0.max() << "\","
                  << "\"minGradientLayer1\": \"" << gradient0.min() << "\",";
      }

      if (!gradient1.is_empty())
      {
        std::cout << "\"maxGradientLayer3\": \"" << gradient1.max() << "\","
                  << "\"minGradientLayer3\": \"" << gradient1.min() << "\",";
      }

      for (size_t i = 0; i < 11; i++)
        std::cout << "\"predBlock" << i << "\": \"" << GetPredictionBlock(i) << "\",";

      std::cout << "\"maxActivationLayer0\": \"" << inputLayerPtr->InputActivation().max() << "\","
                << "\"minActivationLayer0\": \"" << inputLayerPtr->InputActivation().min() << "\","
                << "\"maxActivationLayer1\": \"" << convLayer0Ptr->InputActivation().max() << "\","
                << "\"minActivationLayer1\": \"" << convLayer0Ptr->InputActivation().min() << "\","
                << "\"maxActivationLayer2\": \"" << poolingLayer0Ptr->InputActivation().max() << "\","
                << "\"minActivationLayer2\": \"" << poolingLayer0Ptr->InputActivation().min() << "\","
                << "\"maxActivationLayer3\": \"" << convLayer1Ptr->InputActivation().max() << "\","
                << "\"minActivationLayer3\": \"" << convLayer1Ptr->InputActivation().min() << "\","
                << "\"maxActivationLayer4\": \"" << poolingLayer1Ptr->InputActivation().max() << "\","
                << "\"minActivationLayer4\": \"" << poolingLayer1Ptr->InputActivation().min() << "\","
                << "\"maxActivationLayer5\": \"" << outputLayerPtr->InputActivation().max() << "\","
                << "\"minActivationLayer5\": \"" << outputLayerPtr->InputActivation().min() << "\""
                << "}";

      trainingSeqCur = trainingSeqStart;
    }

    /*
     * The prediction matrix using of the specified layer id.
     *
     * @param id The id of the layer.
     * @param m The matrix used to store the prediction.
     */
    void GetPredictionMatrix(int id, arma::mat& m)
    {
      if (id >= 0 && id <= 10)
      {
        if (predictionInfo[id].image == 0 || predictionInfo[id].image->is_empty())
        {
          m = arma::mat(1, 1);
        }
        else
        {
          m = *predictionInfo[id].image;
        }
      }
      else
      {
        m = arma::mat(1, 1);
      }
    }

    //! Get the state.
    int State() const { return state; }
    //! Modify the state.
    int& State() { return state; }

  private:
    struct PredictionInfo
    {
      arma::mat* image = 0;
      std::string prob[5];
      std::string prediction[5];
    };

    /*
     * Helper function to return the prediction.
     *
     * @param id The id of the layer.
     */
    std::string GetPredictionBlock(int id)
    {
        std::string output = predictionInfo[id].prediction[0] + " - " + predictionInfo[id].prob[0] + " <br>";
        output += predictionInfo[id].prediction[1] + " - " + predictionInfo[id].prob[1] + " <br>";
        output += predictionInfo[id].prediction[2] + " - " + predictionInfo[id].prob[2] + " <br>";
        output += predictionInfo[id].prediction[3] + " - " + predictionInfo[id].prob[3] + " <br>";
        output += predictionInfo[id].prediction[4] + " - " + predictionInfo[id].prob[4];
        return output;
    }

    // The job state (running = 0, queued = 1, stopped = 2).
    int state;

    // The current error and the overall training error.
    double trainingError, trainingErrorcurrent;

    // The current training sequence and the overall training sequence.
    size_t trainingSeqStart, trainingSeqCur;

    // The prediction info (matrix, probability and prediction).
    PredictionInfo predictionInfo[11];

    // The jon content.
    std::string content;

    // The dataset and the target.
    arma::mat X, Y;

    // The number of points in the dataset.
    arma::uword nPoints;

    // The gradient of the first  and second convolution layer.
    arma::cube gradient0, gradient1;

    // The network structure (layer and connections).
    NeuronLayer<RectifierFunction, arma::cube>* inputLayerPtr;

    ConvLayer<RectifierFunction>* convLayer0Ptr;

    ConvConnection<NeuronLayer<RectifierFunction, arma::cube>,
               ConvLayer<RectifierFunction>,
               mlpack::ann::AdaDelta>* con1Ptr;

    BiasLayer<>* biasLayer0Ptr;

    BiasConnection<BiasLayer<>,
               ConvLayer<RectifierFunction>,
               mlpack::ann::AdaDelta,
               mlpack::ann::ZeroInitialization>* con1BiasPtr;

    PoolingLayer<>* poolingLayer0Ptr;

    PoolingConnection<ConvLayer<RectifierFunction>,
                  PoolingLayer<> >* con2Ptr;

    ConvLayer<RectifierFunction>* convLayer1Ptr;

    ConvConnection<PoolingLayer<>,
               ConvLayer<RectifierFunction>,
               mlpack::ann::AdaDelta>* con3Ptr = 0;

    BiasLayer<>* biasLayer3Ptr;

    BiasConnection<BiasLayer<>,
                     ConvLayer<RectifierFunction>,
                     mlpack::ann::AdaDelta,
                     mlpack::ann::ZeroInitialization>* con3BiasPtr;

    PoolingLayer<>* poolingLayer1;

    PoolingConnection<ConvLayer<RectifierFunction>,
                      PoolingLayer<> >* con4;

    SoftmaxLayer<arma::mat>* outputLayerPtr;

    FullConnection<PoolingLayer<>,
                   SoftmaxLayer<arma::mat>,
                   mlpack::ann::AdaDelta>* con5Ptr;

    BiasLayer<>* biasLayer1Ptr;

    FullConnection<BiasLayer<>,
               SoftmaxLayer<arma::mat>,
               mlpack::ann::AdaDelta,
               mlpack::ann::ZeroInitialization>* con5BiasPtr;

    MulticlassClassificationLayer* finalOutputLayerPtr;

    PoolingLayer<>* poolingLayer1Ptr;

    PoolingConnection<ConvLayer<RectifierFunction>,
                        PoolingLayer<> >* con4Ptr;
};

/*
 * Job wrapper class that holds the actual job.
 */
class JobWrapper {
  public:
    /*
     * Create the JobWrapper object and store the specified job.
     */
    JobWrapper(Job& job) : job(&job)
    {
      /* Nothing to do here */
    }

    /*
     * Dispatch the job.
     */
    void operator()(){
      job->Run();
      return;
    }

  public:
    //! Locally stored pointer of an job object.
    Job* job;
};

/*
 * Redirect the user to the initial page.
 *
 * @param id The user/job id.
 */
void InitNewPage(const int id)
{
  std::cout << "Content-type: text/html\r\n\r\n"
            << "<meta http-equiv=\"refresh\" content=\"0;"
            << " URL=" + config::url + "/id=" << id << "\" />";
}

/*
 * Create image tag using the specified informations.
 *
 * @param id The user/job id.
 * @param imageNumber The number of the image, unique identifier.
 * @param name The name of the image event.
 * @param width The image width.
 * @param height The height width.
 */
void CreateImageTag(const int id,
                    const int imageNumber,
                    const std::string name,
                    const int width,
                    const int height)
{
  std::cout << "<img width=\"" << width << "\" height=\"" << height
            << "\" src=\"" << config::path << "/id=" << std::to_string(id)
            << "&" << name << "=" << std::to_string(imageNumber)
            << "&rnd=23" << "\" />\n";
}

/*
 * Create prediction block using the specified informations.
 *
 * @param id The user/job id.
 * @param width The image width.
 * @param height The height width.
 * @param imageNumber The number of the image, unique identifier.
 * @param name The name of the image event.
 */
void AddPredictionBlock(const int id,
                        const int width,
                        const int height,
                        const int imageNumber,
                        const std::string name)
{
  std::cout << "<div class=\"probsdiv col l1 m2 s3\">"
            << "<img style=\"display: block; margin: 0 auto;\" width=\""
            << width << "\" height=\"" << height
            << "\" src=\"" << config::path << "/id=" << std::to_string(id)
            << "&" << name << "=" << std::to_string(imageNumber) << "&rnd=23"
            << "\" />\n<div>"
            << "<span id=\"predBlock" << imageNumber << "\"></span>"
            << "</div></div>";
}

/*
 * Start prediction container. The prediction container can hold a various
 * prediction informations.
 */
void StartPredictionContainer()
{
  std::cout << "<div class=\"container\">"
            << "<div class=\"col s12 m6\">"
            << "<div class=\"card blue-grey darken-1\">"
            << "<div class=\"card-content white-text\">"
            << "<span class=\"card-title\">Example predictions</span>"
            << "<div class=\"testset row\">";
}

/*
 * Stop the prediction container. The prediction container can hold a various
 * prediction informations.
 */
void StopPredictionContainer()
{
  std::cout << "</div></div></div></div></div>";
}

/*
 * Start layer container. The layer container can hold a various
 * layer informations including images.
 *
 * @param name The name of the layer.
 * @param id The user/job id.
 * @param textNumber The number of the layer, unique identifier.
 * @param gradient Enables the gradient images.
 * @param weight Enables the weight images.
 */
void StartLayerContainer(const std::string name,
                         const int id,
                         const int textNumber,
                         bool gradient = false,
                         bool weight = false)
{
  std::cout << "<div class=\"container\">"
            << "<div class=\"col s12 m6\">"
            << "<div class=\"card teal darken-1\">"
            << "<div class=\"card-content white-text\">"
            << "<div class=\"row\">"
            << "<div class=\"col s4\">"
            << "<h6>" << name << "</h6>"
            << "<span>max activation: "
            << "<span id=\"maxActivationLayer" << textNumber << "\"></span>"
            << ", min: "
            << "<span id=\"minActivationLayer" << textNumber << "\"></span>"
            << "</span>";

  if (weight)
  {
    std::cout << "<br>"
              << "<span>max weight: "
              << "<span id=\"maxWeightLayer" << textNumber << "\"></span>"
              << ", min: "
              << "<span id=\"minWeightLayer" << textNumber << "\"></span>"
              << "</span>";
  }

  if (gradient)
  {
    std::cout << "<br>"
              << "<span>max gradient: "
              << "<span id=\"maxGradientLayer" << textNumber << "\"></span>"
              << ", min: "
              << "<span id=\"minGradientLayer" << textNumber << "\"></span>"
              << "</span>";
  }


  std::cout << "</div><div class=\"col s8\">";
}

/*
 * Start inner layer container. The inner layer container holds the images and
 * layer informations.
 *
 * @param name The name of the layer information (e.g. Gradients, Weights or
 * Activations).
 */
void StartInnerContainer(const std::string name)
{
  std::cout << "<div class=\"container__inner\">" << "<h6>" << name << ":</h6>";
}

/*
 * Stop the inner layer container.
 */
void StopInnerContainer()
{
  std::cout << "</div>";
}

/*
 * Stop the main layer container.
 */
void EndLayerContainer()
{
  std::cout <<  "</div></div></div></div></div></div>";
}

/*
 * Update the main page including all javascript and chart functions.
 *
 * @param id The user/job id.
 */
void UpdatePage(const int id)
{
  std::cout << "Content-type: text/html\r\n"
            << "\r\n"
            << "<!doctype html>"
            << "<html lang=\"en\">"
            << "<head>"
            << "<meta charset=\"UTF-8\">"
            << "<title>MNIST demo</title>"
            << "<link rel=\"stylesheet\" href=\"styles/css/materialize.min.css\">"
            << "<link rel=\"stylesheet\" href=\"styles/css/style.css\">"
            << "<script src=\"styles/js/Chart.js\"></script>"
            << "<script src=\"styles/js/jquery.min.js\"></script>"
            << "<script language=\"JavaScript\">"

            << "var run = true;"
            << "var view = true;"

            << "function createChart() {"
            << "var data = {"
            << "labels: [\"0\", \"0\"],"
            << "datasets: ["
            << "{"
            << "  label: \"Training Error\","
            << "  fillColor: \"rgba(220,220,220,0.2)\","
            << "  strokeColor: \"rgba(220,220,220,1)\","
            << "  pointColor: \"rgba(220,220,220,1)\","
            << "  pointStrokeColor: \"#fff\","
            << "  pointHighlightFill: \"#fff\","
            << "  pointHighlightStroke: \"rgba(220,220,220,1)\","
            << "  data: [0, 0]"
            << "}"
            <<  "]"
            << "};"

            << "var ctx = document.getElementById(\"trainingChart\").getContext(\"2d\");"
            << "window.myLine = new Chart(ctx).Line(data, {"
            << "responsive: true"
            << "});"
            << "}"


            << "function startStopView() {"
            <<   "if (view == true) {"
            <<    "document.getElementById('viewbtn').innerHTML = \"Enable View Update\";"
            <<    "view = false;"
            <<  "} else {"
            <<    "document.getElementById('viewbtn').innerHTML = \"Disable View Update\";"
            <<    "view = true;"
            <<    "refreshIt();"
            <<  "}"
            << "}"

            << "function startStop() {"
            << "var xmlHttp = new XMLHttpRequest();"
            << "xmlHttp.open(\"GET\", \"" << config::url << "/id=" << id
            << "&start=0\", true);"
            << "xmlHttp.send();"
            <<   "if (run == true) {"
            <<    "document.getElementById('trainingbtn').innerHTML = \"Start Training\";"
            <<    "run = false;"
            <<  "} else {"
            <<    "document.getElementById('trainingbtn').innerHTML = \"Stop Training\";"
            <<    "run = true;"
            <<    "refreshIt();"
            <<  "}"
            << "}"

            << "function State() {"
            <<   "var xmlHttp = new XMLHttpRequest();"
            <<   "xmlHttp.open(\"GET\", \"" << config::url << "/id=" << id
            <<   "&state=0\", false);"
            <<   "xmlHttp.send();"
            <<   "var x = xmlHttp.responseText;"
            <<   "if (typeof x === \"number\" && typeof x === \"string\" || x !== \"\") {"
            <<     "x = Number(x);"
            <<     "if (x == 0) {"
            <<        "document.getElementById('trainingbtn').innerHTML = \"Stop Training\";"
            <<        "run = true;"
            <<     "}"

            <<     "if (x == 2) {"
            <<        "document.getElementById('trainingbtn').innerHTML = \"Start Training\";"
            <<        "run = false;"
            <<     "}"

            <<   "}"
            << "}"

            << "function refreshIt() {"
            << "var xmlHttp = new XMLHttpRequest();"
            << "xmlHttp.open(\"GET\", \"" << config::url << "/id=" << id
            << "&info=0\", false);"
            << "xmlHttp.send();"
            << "jsonObj = JSON.parse(xmlHttp.responseText);"
            << "var jsonKeys = Object.keys(jsonObj);"
            << "for (var i = 0; i < jsonKeys.length; i++) {"
            <<    "var element = document.getElementById(jsonKeys[i]);"
            <<    "if (element != null) {"
            <<        "element.innerHTML = jsonObj[jsonKeys[i]];"
            <<     "} else {"
            <<        "if (jsonKeys[i] == \"trainingError\") {"
            <<           "console.info(jsonObj);"
            <<           "window.myLine.addData([jsonObj[jsonKeys[i]]], \"0\");"
            <<        "}"
            <<     "}"
            << "}"
            << "if (!document.images) return;"
            << "var images = document.images;"
            << "for(var i = 0; i < images.length; i++) {"
            << "var pos = images[i].src.indexOf(\"rnd=\");"
            << "if (pos > 0) {"
            << "images[i].src = images[i].src.substring(0, pos + 4) + Math.random(); } }"
            << "if (run == true && view == true) {"
            <<   "setTimeout('refreshIt()',2000);"
            << "}"
            << "}"
            << "</script>"
            << "</head>"
            << "<body onLoad=\" State(); setTimeout('refreshIt()',1000); createChart();\">"
            <<    "<div class=\"wrapper\">"
            <<        "<h2 class=\"heading center\">MNIST Demo</h2>"
            <<        "<div class=\"container\">"
            <<            "<div class=\"col s12 m6\">"
            <<              "<div class=\"card blue-grey darken-1\">"
            <<                "<div class=\"card-content white-text\">"
            <<                  "<span class=\"card-title\">Description</span>"
            <<                  "<p>This demo trains a Convolutional Neural Network on the <a href=\"http://yann.lecun.com/exdb/mnist/\">MNIST Dataset</a> (Mixed National Institute of Standards and Technology database) digits dataset and shows the training process in your browser. We will use the architecture known as <a href=\"http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf\">LeNet</a>, which is a deep convolutional neural network known to work well on handwritten digit classification tasks.</p>"
            <<                  "<p>More precisely, we will use <a href=\"http://mlpack.org\">mlpack's</a> neural network architecture, to build a modified version of the network by replacing the sigmoid activation functions with Rectified Learning Unit (ReLU) activation functions. The basic network structure consists of a convolution layer followed by a pooling layer, and then another convolution followed by a pooling layer. After that, one densely connected layer is added.</p>"
            <<                "</div>"
            <<              "</div>"
            <<            "</div>"
            <<        "</div>"
            <<        "<div class=\"container\">"
            <<            "<div class=\"col s12 m6\">"
            <<              "<div class=\"card blue-grey darken-1\">"
            <<                "<div class=\"card-content white-text\">"
            <<                    "<span class=\"card-title\">Training Stats</span>"
            <<                        "<div class=\"settings--holder\">"

            << "<div class=\"row\">"
            << "<form class=\"col s12\">"
            <<   "<div class=\"row\">"
            <<      "<div class=\"input-field col s12\">"
            <<        "<a id =\"trainingbtn\" class=\"waves-effect waves-light btn\" onclick=\"startStop()\">Stop Training</a>\n"
            <<        "<a id =\"viewbtn\" class=\"waves-effect waves-light btn\" onclick=\"startStopView()\">Disable View Update</a>\n"
            <<      "<input class=\"with-gap\" name=\"group1\" type=\"radio\" id=\"RMSprop\" disabled=\"disabled\" />\n"
            <<      "<label for=\"RMSprop\">RMSprop</label>\n"
            <<      "<input class=\"with-gap\" name=\"group1\" type=\"radio\" id=\"AdaDelta\" checked />\n"
            <<      "<label for=\"AdaDelta\">AdaDelta</label>\n"
            <<      "<input class=\"with-gap\" name=\"group1\" type=\"radio\" id=\"SGD\" disabled=\"disabled\" />\n"
            <<      "<label for=\"SGD\">SGD</label>\n"
            <<      "<input class=\"with-gap\" name=\"group1\" type=\"radio\" id=\"ADAGrad\" disabled=\"disabled\" />\n"
            <<      "<label for=\"ADAGrad\">ADAGrad</label>\n"
            <<      "</div>"
            <<    "</div>"
            <<  "</form>"
            << "</div>"
            << "</div>"


            <<                        "<div class=\"card-action\">"
            <<                          "<canvas id=\"trainingChart\" height=\"50\"></canvas>"
            <<                        "</div>"
            <<                "</div>"
            <<            "</div>"
            <<        "</div>"
            <<        "</div>";

            // Create the page structure.
            StartLayerContainer("input (24x24x1):", id, 0);
            StartInnerContainer("Activation");
            CreateImageTag(id, 0, "input", 48, 48);
            StopInnerContainer();
            EndLayerContainer();

            StartLayerContainer("convolution (24x24x8)", id, 1, true, true);
            StartInnerContainer("Activations");
            for (int i = 1; i < 7; i++)
              CreateImageTag(id, i, "input", 44, 44);
            StopInnerContainer();
            StartInnerContainer("Weights");
            for (int i = 0; i < 6; i++)
              CreateImageTag(id, i, "weight", 30, 30);
            StopInnerContainer();
            StartInnerContainer("Gradients");
            for (int i = 0; i < 6; i++)
              CreateImageTag(id, i, "gradient", 30, 30);
            StopInnerContainer();
            EndLayerContainer();

            StartLayerContainer("pooling (24x24x1):", id, 2);
            StartInnerContainer("Activations");
            for (int i = 7; i < 13; i++)
              CreateImageTag(id, i, "input", 30, 30);
            StopInnerContainer();
            EndLayerContainer();

            StartLayerContainer("convolution (24x24x1):", id, 3, true, true);
            StartInnerContainer("Activations");
            for (int i = 13; i < 23; i++)
              CreateImageTag(id, i, "input", 30, 30);
            StopInnerContainer();
            StartInnerContainer("Weights");
            for (int i = 6; i < 66; i++)
              CreateImageTag(id, i, "weight", 20, 20);
            StopInnerContainer();
            StartInnerContainer("Gradients");
            for (int i = 6; i < 66; i++)
              CreateImageTag(id, i, "gradient", 20, 20);
            StopInnerContainer();
            EndLayerContainer();

            StartLayerContainer("pooling (24x24x1):", id, 4);
            StartInnerContainer("Activations");
            for (int i = 23; i < 33; i++)
              CreateImageTag(id, i, "input", 15, 15);
            StopInnerContainer();
            EndLayerContainer();

            StartLayerContainer("softmax (24x24x1):", id, 5);
            StartInnerContainer("Activations");
            for (int i = 33; i < 43; i++)
              CreateImageTag(id, i, "input", 15, 15);
            StopInnerContainer();
            EndLayerContainer();

  StartPredictionContainer();
  for (size_t i = 0; i < 11; i++)
    AddPredictionBlock(id, 48, 48, i, "prediction");
  StopPredictionContainer();

  std::cout << "</div>"
            << "</div>"
            << "<script src=\"styles/js/materialize.min.js\"></script>"
            << "</body>"
            << "</html>";
}


int main(int argc, char **argv)
{
  // Load the dataset.
  arma::mat X;
  X.load(config::dataset);

  boost::ptr_vector<Job> jobQueue;

  boost::asio::io_service io_service;
  boost::thread_group threads;
  std::auto_ptr<boost::asio::io_service::work> work(
      new boost::asio::io_service::work(io_service));

  // Spawn worker threads.
  size_t cores = std::thread::hardware_concurrency();
  for (size_t i = 0; i < cores; i++)
  {
    threads.create_thread(boost::bind(
        &boost::asio::io_service::run, &io_service));
  }

  // Backup the stdio streambufs.
  std::streambuf* cinStreambuf  = std::cin.rdbuf();
  std::streambuf* coutStreambuf = std::cout.rdbuf();

  FCGX_Request request;

  FCGX_Init();
  FCGX_InitRequest(&request, 0, 0);

  size_t counter = 0;

  // Handle requests.
  while (FCGX_Accept_r(&request) == 0) {

      fcgi_streambuf cin_fcgi_streambuf(request.in);
      fcgi_streambuf cout_fcgi_streambuf(request.out);

      std::cin.rdbuf(&cin_fcgi_streambuf);
      std::cout.rdbuf(&cout_fcgi_streambuf);

      const char* uri = FCGX_GetParam("REQUEST_URI", request.envp);
      std::string content = http::RequestContent(request);

      // Print the request uri.
      std::cerr << uri << std::endl;

      // Parse the request uri and extract the event and the event id.
      parser::info uriInfo;
      parser::parseURI(uri, uriInfo);

      // Handle the event.
      if (uriInfo.id < 0 || (uriInfo.id + 1) > jobQueue.size() )
      {
        // Create a new task.
        jobQueue.push_back(new Job(uri, X));
        io_service.dispatch(boost::move(JobWrapper(jobQueue.back())));

        // Redirect the user to the initial page using the new job id.
        InitNewPage(jobQueue.size() - 1);
      }
      else
      {
        if (uriInfo.event == "")
        {
          UpdatePage(uriInfo.id);
        }
        else if(uriInfo.event == "input" && uriInfo.eventId >= 0)
        {
          // Send the requested image.
          arma::mat m;
          jobQueue[uriInfo.id].GetInputMatrix(uriInfo.eventId, m);
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << graphics::Mat2Image(m);
        }
        else if(uriInfo.event == "start" && uriInfo.eventId >= 0)
        {
          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the jon, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }
          else if (state == 2)
          {
            // Queue the job to be running next.
            jobQueue[uriInfo.id].State() = 1;
            io_service.dispatch(boost::move(JobWrapper(jobQueue[uriInfo.id])));
          }

          // Send the new job state.
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << jobQueue[uriInfo.id].State();
        }
        else if(uriInfo.event == "state" && uriInfo.eventId >= 0)
        {
          // Send the job state.
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << jobQueue[uriInfo.id].State();
        }
        else if(uriInfo.event == "weight" && uriInfo.eventId >= 0)
        {
          // Send the weight image.
          arma::mat m;
          jobQueue[uriInfo.id].GetWeightMatrix(uriInfo.eventId, m);
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << graphics::Mat2Image(m);
        }
        else if(uriInfo.event == "gradient" && uriInfo.eventId >= 0)
        {
          // Send the gradient image.
          arma::mat m;
          jobQueue[uriInfo.id].GetGradientMatrix(uriInfo.eventId, m);
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << graphics::Mat2Image(m);
        }
        else if(uriInfo.event == "prediction" && uriInfo.eventId >= 0)
        {
          // Send the prediction image.
          arma::mat m;
          jobQueue[uriInfo.id].GetPredictionMatrix(uriInfo.eventId, m);
          std::cout << "Content-Type:image/jpeg\r\n\r\n";
          std::cout << graphics::Mat2Image(m);
        }
        else if(uriInfo.event == "info" && uriInfo.eventId >= 0)
        {
          // Send all information about the job as encoded json string.
          std::cout << "Content-type: text/html\r\n\r\n";
          jobQueue[uriInfo.id].GetInfoText(uriInfo.eventId);
        }
      }
  }

  // Waiting for the worker threads to finish the last task.
  work.reset();

  // Restore stdio streambufs.
  std::cin.rdbuf(cinStreambuf);
  std::cout.rdbuf(coutStreambuf);

  return 0;
}
