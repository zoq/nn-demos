/**
 * @file job.hpp
 * @author Marcus Edel
 *
 * Miscellaneous job settings.
 */

#include <iostream>
#include <mutex>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/asio.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/base64/base64.hpp>

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

#include "../parser.hpp"
#include "../graphics.hpp"
#include "config.hpp"

namespace job {

  std::mutex jobMutex;

  using namespace mlpack;
  using namespace mlpack::ann;

  // Additional data for every connection.
  struct connection_data {
    int sessionid;
  };

  // Custom config to manage session ids.
  struct custom_config : public websocketpp::config::asio {
    // Pull default settings from our core config.
    typedef websocketpp::config::asio core;

    typedef core::concurrency_type concurrency_type;
    typedef core::request_type request_type;
    typedef core::response_type response_type;
    typedef core::message_type message_type;
    typedef core::con_msg_manager_type con_msg_manager_type;
    typedef core::endpoint_msg_manager_type endpoint_msg_manager_type;
    typedef core::alog_type alog_type;
    typedef core::elog_type elog_type;
    typedef core::rng_type rng_type;
    typedef core::transport_type transport_type;
    typedef core::endpoint_base endpoint_base;

    // Set a custom connection_base class.
    typedef connection_data connection_base;
  };

  typedef websocketpp::server<custom_config> server;
  typedef server::connection_ptr connection_ptr;

  using websocketpp::connection_hdl;
  using websocketpp::lib::placeholders::_1;
  using websocketpp::lib::placeholders::_2;
  using websocketpp::lib::bind;

  /*
   * Class to handle the training process of the specified neural network.
   */
  class JobContainer {
    public:
      /*
       * Create the job object to train the neural network on the specified
       * dataset.
       */
      JobContainer(server& s,
                   connection_hdl hdl,
                   int sessionid,
                   arma::mat& XTrain,
                   arma::mat& YTrain,
                   arma::mat& XTest,
                   arma::mat& YTest) :
        XTrain(XTrain),
        YTrain(YTrain),
        XTest(XTest),
        YTest(YTest),
        s(&s),
        sessionid(sessionid)
      {
        nPointsTrain = XTrain.n_cols;
        nPointsTest = XTest.n_cols;

        // Add connection handle to the list of all connections. We use the list
        // to send all connected clients the data generated through the training
        // process.
        connectionHandles.push_back(hdl);

        // Create the network structure.
        inputLayerPtr = new NeuronLayer<RectifierFunction,
            arma::cube>(28, 28, 1);

        convLayer0Ptr = new ConvLayer<RectifierFunction>(24, 24,
            inputLayerPtr->LayerSlices(), 6);

        con1Ptr = new ConvConnection<NeuronLayer<RectifierFunction, arma::cube>,
                                                 ConvLayer<RectifierFunction>,
                                                 mlpack::ann::AdaDelta>
                                                 (*inputLayerPtr,
                                                  *convLayer0Ptr, 5);

        biasLayer0Ptr = new BiasLayer<>(6);

        con1BiasPtr = new BiasConnection<BiasLayer<>,
                                         ConvLayer<RectifierFunction>,
                                         mlpack::ann::AdaDelta,
                                         mlpack::ann::ZeroInitialization>
                                         (*biasLayer0Ptr, *convLayer0Ptr);

        poolingLayer0Ptr = new PoolingLayer<>(12, 12,
            inputLayerPtr->LayerSlices(), 6);

        con2Ptr = new PoolingConnection<ConvLayer<RectifierFunction>,
                                        PoolingLayer<> >
                                        (*convLayer0Ptr, *poolingLayer0Ptr);

        convLayer1Ptr = new ConvLayer<RectifierFunction>(8, 8,
            inputLayerPtr->LayerSlices(), 10);


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

        poolingLayer1Ptr  = new PoolingLayer<>(4, 4,
            inputLayerPtr->LayerSlices(), 10);

        con4Ptr = new PoolingConnection<ConvLayer<RectifierFunction>,
                                        PoolingLayer<> >
                                        (*convLayer1Ptr, *poolingLayer1Ptr);

        outputLayerPtr = new SoftmaxLayer<arma::mat>(10,
            inputLayerPtr->LayerSlices());

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

        // Initilize the gradient storage.
        gradient0 = con1Ptr->Weights();
        gradient1 = con3Ptr->Weights();

        trainingError = 0;
        trainingErrorcurrent = 0;
        trainingSeqCur = 0;
        trainingSeqStart = 0;
        state = 1; // Waiting

        // Send current job state.
        SendState();
      }

      /*
       * Run the job.
       */
      void Run()
      {
        jobState = 0;
        state = 0; // Running

        SendState();

        auto module0 = std::tie(*con1Ptr, *con1BiasPtr);
        auto module1 = std::tie(*con2Ptr);
        auto module2 = std::tie(*con3Ptr, *con3BiasPtr);
        auto module3 = std::tie(*con4Ptr);
        auto module4 = std::tie(*con5Ptr, *con5BiasPtr);
        auto modules = std::tie(module0, module1, module2, module3, module4);

        CNN<decltype(modules), decltype(*finalOutputLayerPtr)>
        net(modules, *finalOutputLayerPtr);

        Trainer<decltype(net)> trainer(net, 1, 4, 0.03);

        arma::cube inputPrediction = arma::cube(28, 28, 1);
        arma::mat predictionOutput;
        arma::Col<size_t> indexTrain = arma::linspace<arma::Col<size_t> >(0,
            nPointsTrain - 1, nPointsTrain);

        arma::Col<size_t> indexTest = arma::linspace<arma::Col<size_t> >(0,
            nPointsTest - 1, nPointsTest);

        arma::mat error;
        int batchSize = 2;
        bool predict = true;

        bool predictOnly = true;
        for (size_t z = 0; z < 10; z++)
        {
          bool predictOnly = true;
          SendState();

          if (state == 2)
          {
            break;
          }

          if ((z % 10) == 0)
            trainingSeqStart++;

          trainingError = 0;
          testingError = 0;
          predict = true;
          for (size_t i = 0; i < nPointsTrain; i++)
          {
            SendTrainInfo(z, i);

            if (state == 2)
            {
              break;
            }

            if ((i % 500) == 0)
              predict = true;

            indexTrain = arma::shuffle(indexTrain);

            arma::cube inputTemp = arma::cube(28, 28, 1);
            inputTemp.slice(0) = arma::mat(XTrain.colptr(indexTrain(i)), 28, 28);

            arma::mat targetTemp = YTrain.col(indexTrain(i));
            net.FeedForward(inputTemp,  targetTemp, error);

            if (predict || ((i % 250) == 0))
            {
              SendActivation(inputTemp.slice(0), 0);
              SendActivation(convLayer0Ptr->InputActivation(), 1);
              SendActivation(poolingLayer0Ptr->InputActivation(), 7);
              SendActivation(convLayer1Ptr->InputActivation(), 13);
              SendActivation(poolingLayer1Ptr->InputActivation(), 23);

              arma::colvec outputTemp = outputLayerPtr->InputActivation().col(0);
              SendActivation(outputTemp, 33);

              SendWeight(con1Ptr->Weights(), 1);
              SendWeight(con3Ptr->Weights(), 7);
            }

            trainingError += net.Error();

            net.FeedBackward(error);

            if (!con1Ptr->Optimizer().Gradient().is_empty() &&
                (con1Ptr->Optimizer().Gradient().max() > 0))
            {
              gradient0 = arma::cube(con1Ptr->Optimizer().Gradient());
            }

            if (!con3Ptr->Optimizer().Gradient().is_empty() &&
                (con3Ptr->Optimizer().Gradient().max() > 0))
            {
              gradient1 = arma::cube(con3Ptr->Optimizer().Gradient());
            }


            if (predict || ((i % 250) == 0))
            {
              SendGradient(gradient0, 1);
              SendGradient(gradient1, 7);
            }

            if (((i + 1) % batchSize) == 0)
            {
              net.ApplyGradients();
            }


            if (predict)
            {
              predict = false;
              indexTest = arma::shuffle(indexTest);
              try {

                for (size_t k = 0; k < nPointsTest; k++)
                {
                  arma::mat* tempTestInput = new arma::mat(
                      XTest.colptr(indexTest(k)),28, 28);

                  inputPrediction.slice(0) = *tempTestInput;
                  net.Predict(inputPrediction, predictionOutput);

                  if (predictOnly)
                  {
                    testingError += net.Error();
                  }

                  if (!predictionOutput.is_finite())
                    continue;

                  // Collection data for the prediction block.
                  if (k < 11)
                  {
                    predictionInfo[k].image = tempTestInput;
                    arma::uvec idx = arma::sort_index(predictionOutput.col(0),
                        "descend");
                    for (size_t j = 0; j < 5; j++)
                    {
                      predictionInfo[k].prediction[j] = std::to_string(idx(j));
                      predictionInfo[k].prob[j] = std::to_string(
                          predictionOutput(idx(j), 0));
                    }

                    arma::uword indexMax;
                    YTest.col(indexTest(k)).max(indexMax);
                    predictionInfo[k].target = std::to_string(indexMax);
                  }
                  else if (!predictOnly || z == 0)
                  {
                    break;
                  }
                }

                // Send collected prediction data to all connected clients.
                SendPrediction();

                if (z != 0)
                  predictOnly = false;
              }
              catch(...)
              {
                std::cout << "error: sort_index(): detected non-finite values \n";
              }
            }
          }

          trainingError /= nPointsTrain;
          testingError /= nPointsTest;

          trainingErrorcurrent = trainingError;
          testingErrorcurrent = testingError;


          SendTrainingStatus(z);


        }

        state = 2; // Stop

        SendState();

        jobState = 1;
      }

      //! Get the state.
      int State() const { return state; }
      //! Modify the state.
      int& State() { return state; }

      /*
       * Add a connection handle to the list of all connections managed by this
       * job. We use the connection list to send the training state to all
       * connected clients.
       */
      void AddConnectionHandle(connection_hdl hdl)
      {
        connectionHandles.push_back(hdl);
        SendState();
      }

      /*
       * Send the current state to all connected clients.
       */
      void SendState()
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"state\": \"" + std::to_string(state) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      int jobState;

    private:
      /*
       * Send the current training and testing info to all connected clients.
       */
      void SendTrainInfo(size_t iteration, size_t sampleIndex)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream trainingErrorString;
        trainingErrorString << std::fixed << std::setprecision(2);
        trainingErrorString << trainingErrorcurrent;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"sampleIndex\": \"" + std::to_string(sampleIndex) + "\",";
        output += "\"sampleIndex\": \"" + std::to_string(sampleIndex) + "\",";
        output += "\"classificationLoss\": \"" + trainingErrorString.str() + "\",";
        output += "\"iteration\": \"" + std::to_string(iteration) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current training state to all connected clients.
       */
      void SendTrainingStatus(size_t z)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream trainingErrorString;
        trainingErrorString << std::fixed << std::setprecision(2);
        trainingErrorString << trainingErrorcurrent;

        std::stringstream testingErrorString;
        testingErrorString << std::fixed << std::setprecision(2);
        testingErrorString << testingErrorcurrent;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"trainingError\": \"" + trainingErrorString.str() + "\",";
        output += "\"testingError\": \"" + testingErrorString.str() + "\",";
        output += "\"iteration\": \"" + std::to_string(z) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current prediction to all connected clients.
       */
      void SendPrediction()
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < 11; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(*predictionInfo[i].image) * 255);
          output += "\"predictionActivation" + std::to_string(i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";

          output += "\"predictionValues" + std::to_string(i) + "\": \"";
          std::stringstream predictionValues;
          predictionValues << std::fixed << std::setprecision(2);
          for (size_t j = 0; j < 5; j++)
          {
            predictionValues << predictionInfo[i].prediction[j] << "-";
          }
          predictionValues << predictionInfo[i].target;
          output += predictionValues.str() + "\",";

          output += "\"probValues" + std::to_string(i) + "\": \"";
          std::stringstream probValues;
          probValues << std::fixed << std::setprecision(2);
          for (size_t j = 0; j < 5; j++)
          {
            probValues << predictionInfo[i].prob[j] << "-";
          }
          probValues << predictionInfo[i].target;
          output += probValues.str() + "\",";
        }

        output += "\"state\": \"" + std::to_string(state) + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::mat& m, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        std::string imageString = graphics::Mat2Image(arma::normalise(m) * 255);
        output += "\"layerActivation" + std::to_string(id) + "\": \"";
        output += websocketpp::base64_encode(imageString) + "\",";

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << m.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << m.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::colvec& v, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < v.n_elem; i++)
        {
          arma::mat vTemp(1, 1);
          vTemp.fill(v(i));
          std::string imageString = graphics::Mat2Image(vTemp * 255);
          output += "\"layerActivation" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << v.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << v.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);
          output += "\"layerActivation" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current weight to all connected clients.
       */
      void SendWeight(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);

          output += "\"layerWeight" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxWeight" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minWeight" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current gradient to all connected clients.
       */
      void SendGradient(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);

          output += "\"layerGradient" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxGradient" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minGradient" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      // Structure used to manage the predictions.
      struct PredictionInfo
      {
        arma::mat* image = 0;
        std::string prob[5];
        std::string prediction[5];
        std::string target;
      };

      // The job state (running = 0, queued = 1, stopped = 2).
      int state;

      // The current error and the overall training error.
      double trainingError, trainingErrorcurrent;

      // The current error and the overall testing error.
      double testingError, testingErrorcurrent;

      // The current training sequence and the overall training sequence.
      size_t trainingSeqStart, trainingSeqCur;

      // The prediction info (matrix, probability and prediction.
      PredictionInfo predictionInfo[11];

      // The jon content.
      std::string content;

      // The dataset and the target.
      arma::mat XTrain, YTrain, XTest, YTest;

      // The number of points in the dataset.
      arma::uword nPointsTrain, nPointsTest;

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

      //! Locally stored server object used for sending.
      server* s;

      //! List of connection handles.
      std::vector<connection_hdl> connectionHandles;

      //! Locally stored session id.
      int sessionid;

      bool predictionAvailable;
};

  /*
   * Redirect the user to the initial page.
   *
   * @param id The user/job id.
   */
  std::string InitNewPage(const int id)
  {
    std::string output = "{";
    output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
    output += "\"redirect\": \"" + config::url + "?id=";
    output += std::to_string(id) + "\"";
    output += "}";

    return output;
  }

  /*
   * Job wrapper class that holds the actual job.
   */
  class JobWrapper {
    public:
      /*
       * Create the JobWrapper object and store the specified job.
       */
      JobWrapper(JobContainer& jobContainer) : jobContainer(&jobContainer)
      {
        /* Nothing to do here */
      }

      /*
       * Dispatch the job.
       */
      void operator()(){
        jobContainer->Run();
        return;
      }

    public:
      //! Locally stored pointer of an job object.
      JobContainer* jobContainer;
  };

  boost::ptr_vector<JobContainer> jobQueue;
  boost::asio::io_service io_service;

  class Server {
    public:
      Server() : m_next_sessionid(1) {
        m_server.init_asio();

        m_server.set_open_handler(bind(&Server::on_open,this,_1));
        m_server.set_close_handler(bind(&Server::on_close,this,_1));
        m_server.set_message_handler(bind(&Server::on_message,this,_1,_2));
        m_server.set_access_channels(websocketpp::log::alevel::none);
        m_server.set_error_channels(websocketpp::log::alevel::fail);

        // Load the train dataset.
        XTrain.load(config::trainDataset);

        if (XTrain.n_cols >= 784 && XTrain.n_cols <= 785)
        {
          XTrain = XTrain.t();
        }

        nPointsTrain = XTrain.n_cols;

        // Build the target matrix for the train dataset.
        YTrain = arma::zeros<arma::mat>(10, nPointsTrain);
        for (size_t i = 0; i < nPointsTrain; i++)
        {
          size_t target = XTrain(0, i);
          YTrain.col(i)(target) = 1;
        }

        XTrain.shed_row(0);

        // Normalize each point since these are images of the train dataset.
        for (arma::uword i = 0; i < nPointsTrain; i++)
        {
          XTrain.col(i) /= norm(XTrain.col(i), 2);
        }

        // Load the test dataset.
        XTest.load(config::testDataset);

        if (XTest.n_cols >= 784 && XTest.n_cols <= 785)
        {
          XTest = XTest.t();
        }

        nPointsTest = XTest.n_cols;

        // Build the target matrix for the test dataset.
        YTest = arma::zeros<arma::mat>(10, nPointsTest);
        for (size_t i = 0; i < nPointsTest; i++)
        {
          size_t target = XTest(0, i);
          YTest.col(i)(target) = 1;
        }

        XTest.shed_row(0);

        // Normalize each point since these are images of the test dataset.
        for (arma::uword i = 0; i < nPointsTest; i++)
        {
          XTest.col(i) /= norm(XTest.col(i), 2);
        }
      }

      void on_open(connection_hdl hdl) {
        connection_ptr con = m_server.get_con_from_hdl(hdl);

        con->sessionid = m_next_sessionid++;
      }

      void on_close(connection_hdl hdl) {

          connection_ptr con = m_server.get_con_from_hdl(hdl);
      }

      void on_message(connection_hdl hdl, server::message_ptr msg) {
        connection_ptr connection = m_server.get_con_from_hdl(hdl);


        const std::string& host = connection->get_host();
        const std::string& resource = connection->get_resource();

        std::cout << "host: " << host << " resource: " << resource << std::endl;

        // Parse the request resource and extract the event and the event id.
        parser::info uriInfo;
        parser::parseURI(resource, uriInfo);

        parser::info eventInfo;
        parser::parseURI(msg->get_payload(), eventInfo);

        std::cout << "id: " << uriInfo.id << std::endl;
        std::cout << "event: " << eventInfo.event << std::endl;
        std::cout << "event-id: " << eventInfo.eventId << std::endl;

        // Handle the event.
        if (uriInfo.id < 0 || (uriInfo.id + 1) > jobQueue.size() )
        {
          // Create a new task.
          jobQueue.push_back(new JobContainer(m_server, hdl,
              connection->sessionid, XTrain, YTrain, XTest, YTest));
          io_service.dispatch(boost::move(
              JobWrapper(job::jobQueue.back())));

          // Redirect the user to the initial page using the new job id.
          m_server.send(hdl, InitNewPage(jobQueue.size() - 1),
              websocketpp::frame::opcode::TEXT);
        }
        else if(eventInfo.event == "start" && eventInfo.eventId >= 0)
        {
          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }
          else if (state == 2)
          {
            // Queue the job to be running next.
            jobQueue[uriInfo.id].State() = 1;

            if (jobQueue[uriInfo.id].jobState == 1)
            {
              io_service.dispatch(boost::move(JobWrapper(jobQueue[uriInfo.id])));
            }
            else
            {
              jobQueue[uriInfo.id].State() = 0;
            }
          }

          jobQueue[uriInfo.id].SendState();
        }
        else
        {
          jobQueue[uriInfo.id].AddConnectionHandle(hdl);
          jobQueue[uriInfo.id].SendState();
        }
      }

      void run(uint16_t port) {
        m_server.listen(port);
        m_server.start_accept();
        m_server.run();
      }

    private:
      int m_next_sessionid;
      server m_server;

      arma::mat XTrain, XTest, YTrain, YTest;
      arma::uword nPointsTrain, nPointsTest;
  };
}