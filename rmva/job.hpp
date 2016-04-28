/**
 * @file job.hpp
 * @author Marcus Edel
 *
 * Miscellaneous job settings.
 */

#include <iostream>
#include <mutex>
#include <thread>
#include <chrono>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/asio.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/base64/base64.hpp>

#include <mlpack/core.hpp>

#include <mlpack/methods/rmva/rmva.hpp>

#include <mlpack/methods/ann/layer/glimpse_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/reinforce_normal_layer.hpp>
#include <mlpack/methods/ann/layer/multiply_constant_layer.hpp>
#include <mlpack/methods/ann/layer/constant_layer.hpp>
#include <mlpack/methods/ann/layer/log_softmax_layer.hpp>
#include <mlpack/methods/ann/layer/hard_tanh_layer.hpp>

#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

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
        sessionid(sessionid),
        sendConfusion(false),
        reset(false),
        step(false),
        sampleIndex(0)
      {
        nPointsTrain = XTrain.n_cols;
        nPointsTest = XTest.n_cols;

        // Add connection handle to the list of all connections. We use the list
        // to send all connected clients the data generated through the training
        // process.
        connectionHandles.push_back(hdl);

        const size_t hiddenSize = 256;
        const double unitPixels = 13;
        const double locatorStd = 0.11;
        const size_t imageSize = 28;
        const size_t locatorHiddenSize = 128;
        const size_t glimpsePatchSize = 8;
        const size_t glimpseDepth = 1;
        const size_t glimpseScale = 2;
        const size_t glimpseHiddenSize = 128;
        const size_t imageHiddenSize = 256;
        const size_t numClasses = 10;
        const size_t batchSize = 20;
        rho = 7;
        maxIterations = 500000;

        // Locator network.
        linearLayer0 = new LinearMappingLayer<>(hiddenSize, 2);
        biasLayer0 = new BiasLayer<>(2, 1);
        hardTanhLayer0 = new HardTanHLayer<>();
        reinforceNormalLayer0 = new ReinforceNormalLayer<>(2 * locatorStd);
        hardTanhLayer1 = new HardTanHLayer<>();
        multiplyConstantLayer0 = new MultiplyConstantLayer<>(2 * unitPixels / imageSize);

        // Location sensor network.
        linearLayer1 = new LinearLayer<>(2, locatorHiddenSize);
        biasLayer1 = new BiasLayer<>(locatorHiddenSize, 1);
        rectifierLayer0 = new ReLULayer<>();

        // Glimpse sensor network.
        glimpseLayer0 = new GlimpseLayer<>(1, glimpsePatchSize, glimpseDepth, glimpseScale);
        linearLayer2 = new LinearMappingLayer<>(64, glimpseHiddenSize);
        biasLayer2 = new BiasLayer<>(glimpseHiddenSize, 1);
        rectifierLayer1 = new ReLULayer<>();

        // Glimpse network.
        linearLayer3 = new LinearLayer<>(glimpseHiddenSize + locatorHiddenSize,
            imageHiddenSize);
        biasLayer3 = new BiasLayer<>(imageHiddenSize, 1);
        rectifierLayer2 = new ReLULayer<>();
        linearLayer4 = new LinearLayer<>(imageHiddenSize, hiddenSize);
        biasLayer4 = new BiasLayer<>(hiddenSize, 1);

        // Feedback network.
        recurrentLayer0 = new LinearLayer<>(imageHiddenSize, hiddenSize);
        recurrentLayerBias0 = new BiasLayer<>(hiddenSize, 1);

        // Start network.
        startLayer0 = new AdditionLayer<>(hiddenSize, 1);

        // Transfer network.
        rectifierLayer3 = new ReLULayer<>();

        // Classifier network.
        linearLayer5 = new LinearLayer<>(hiddenSize, numClasses);
        biasLayer6 = new BiasLayer<>(numClasses, 1);
        logSoftmaxLayer0 = new LogSoftmaxLayer<>();

        // Reward predictor network.
        constantLayer0 = new ConstantLayer<>(1, 1);
        additionLayer0 = new AdditionLayer<>(1, 1);

        trainingError = 0;
        trainingErrorcurrent = 0;
        trainingSeqCur = 0;
        trainingSeqStart = 0;
        state = 1; // Waiting

        // Initilize the confusion matrix.
        confusion = arma::Mat<int>(10, 10);

        // Initilize the confusion matrix samples.
        confusionClassification = arma::zeros<arma::Mat<int> >(10 * 10, 25);

        // Initilize the confusion matrix probabilities.
        confusionProps = arma::zeros<arma::mat>(10 * 10, 25);

        // Initilize the confusion matrix.
        confusion = arma::Mat<int>(10, 10);

        // Initilize the confusion matrix samples.
        confusionClassification = arma::zeros<arma::Mat<int> >(10 * 10, 25);

        // Initilize the confusion matrix probabilities.
        confusionProps = arma::zeros<arma::mat>(10 * 10, 25);

        sampleIndex = 0;

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

        arma::Col<size_t> indexTrain = arma::linspace<arma::Col<size_t> >(0,
            nPointsTrain - 1, nPointsTrain);
        indexTrain = arma::shuffle(indexTrain);

        // Locator network.
        auto locator = std::tie(*linearLayer0, *biasLayer0, *hardTanhLayer0,
            *reinforceNormalLayer0, *hardTanhLayer1, *multiplyConstantLayer0);

        // Location sensor network.
        auto locationSensor = std::tie(*linearLayer1, *biasLayer1, *rectifierLayer0);

        // Glimpse sensor network.
        auto glimpseSensor = std::tie(*glimpseLayer0, *linearLayer2, *biasLayer2,
            *rectifierLayer1);

        // Glimpse network.
        auto glimpse = std::tie(*linearLayer3, *biasLayer3, *rectifierLayer2,
            *linearLayer4, *biasLayer4);

        // Feedback network.
        auto feedback = std::tie(*recurrentLayer0, *recurrentLayerBias0);

        // Start network.
        auto start = std::tie(*startLayer0);

        // Transfer network.
        auto transfer = std::tie(*rectifierLayer3);

        // Classifier network.
        auto classifier = std::tie(*linearLayer5, *biasLayer6, *logSoftmaxLayer0);

        // Reward predictor network.
        auto rewardPredictor = std::tie(*constantLayer0, *additionLayer0);

        // Recurrent Model for Visual Attention.
        RecurrentNeuralAttention<decltype(locator),
                                 decltype(locationSensor),
                                 decltype(glimpseSensor),
                                 decltype(glimpse),
                                 decltype(start),
                                 decltype(feedback),
                                 decltype(transfer),
                                 decltype(classifier),
                                 decltype(rewardPredictor),
                                 RandomInitialization>
          net(locator, locationSensor, glimpseSensor, glimpse, start, feedback,
              transfer, classifier, rewardPredictor, rho);

        mlpack::optimization::MiniBatchSGD<decltype(net)> opt(net);
        opt.StepSize() = 0.01;

        opt.MaxIterations() = 2;

        opt.Tolerance() = -200;
        opt.Shuffle() = true;
        opt.BatchSize() = 1;

        // We set the default prediction mode.
        bool predict = true;

        if (sampleIndex == 0 || reset)
        {
          reset = false;
          sampleIndex = 0;
        }

        for (size_t z = 0; z < 100; z++)
        {
          SendState();

          data::Save("model_" + std::to_string(sessionid) + ".xml", "rmva_model", net);

          arma::mat locationInput = XTrain.col(indexTrain(z));
          arma::mat locationTarget;
          locationTarget << YTrain(indexTrain(z));

          SendInput(locationInput);

          net.Train(locationInput, locationTarget, opt);

          arma::mat location = net.Location();
          arma::mat tempInput = arma::zeros(28, 28);
          SendLocation(tempInput, location, 1, 8, 1, 2);

          data::Load("model_" + std::to_string(sessionid) + ".xml", "rmva_model", net);

          if (state == 2)
          {
            break;
          }

          if (!step)
          {
            confusion.zeros();
            confusionClassification.zeros();
            confusionProps.zeros();
          }

          predict = true;

          for (size_t i = 0; i < nPointsTrain; i++)
          {
            sampleIndex++;
            SendTrainInfo(z, sampleIndex);

            if (state == 2)
            {
              break;
            }

            if (((i % 1000) == 0) || step)
              predict = true;

            m.lock();

            arma::mat trainInput = XTrain.col(indexTrain(i));
            arma::mat trainTarget;
            trainTarget << YTrain(indexTrain(i));

            arma::mat trainPredictionOutput;
            net.Predict(trainInput, trainPredictionOutput);

            if (trainPredictionOutput.is_finite())
            {
              arma::uword trainPredictionIndexMax, trainTargetIndexMax;
              trainPredictionOutput.max(trainPredictionIndexMax);

              trainTargetIndexMax = trainTarget(0) - 1;

              confusion(trainTargetIndexMax, trainPredictionIndexMax)++;

              // Get the sample class.
              arma::uword confusionClassIndex = sub2ind(10, 10,
                  trainTargetIndexMax, trainPredictionIndexMax);

              // Get the current sample index.
              arma::uword confusionSampleIndex = confusionClassification(
                  confusionClassIndex, 0);

              // Set the current sample.
              confusionClassification(confusionClassIndex,
                (confusionSampleIndex % 18) + 1) = indexTrain(i);

              // Set the current prediction propability.
              confusionProps(confusionClassIndex,
                  (confusionSampleIndex % 18) + 1) = trainPredictionOutput(
                  trainPredictionIndexMax);

              // Increase the sample index.
              confusionClassification(confusionClassIndex, 0)++;
              confusionProps(confusionClassIndex, 0)++;
            }

            m.unlock();

            if (predict || ((i % 1000) == 0))
            {
              SendConfusion(confusion);

              predict = false;
            }

            if ((i % 15000) == 0)
            {
              data::Save("model_" + std::to_string(sessionid) + ".xml", "rmva_model", net);

              locationInput = XTrain.col(indexTrain(i));
              locationTarget(0) = YTrain(indexTrain(i));

              SendInput(locationInput);

              net.Train(locationInput, locationTarget, opt);

              arma::mat location = net.Location();
              arma::mat tempInput = arma::zeros(28, 28);
              SendLocation(tempInput, location, 1, 8, 1, 2);

              data::Load("model_" + std::to_string(sessionid) + ".xml", "rmva_model", net);
            }

            if (step)
            {
              state = 2; // Stop
              step = false;
              break;
            }
          }

          indexTrain = arma::shuffle(indexTrain);
        }

        state = 2; // Stop

        SendState();

        jobState = 1;
      }

      //! Get the state.
      int State() const { return state; }
      //! Modify the state.
      int& State() { return state; }

      //! Get the reset mode.
      bool Reset() const { return reset; }
      //! Modify the reset mode.
      bool& Reset() { return reset; }

      //! Get the step mode.
      bool Step() const { return step; }
      //! Modify the step mode.
      bool& Step() { return step; }

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
       * Send the confusion samples of the given id to all connected clients.
       */
      void ConfusionSamples(int id)
      {
        m.lock();

        if (confusionClassification.empty())
        {
          m.unlock();
          return;
        }

        sendConfusion = true;
        size_t classIndex = 0;

        if (id <= 9)
        {
          classIndex = sub2ind(10, 10, id, 0);
        }
        else
        {
          std::string idStr = std::to_string(id);
          std::string rowStr(1, idStr.at(1));
          std::string colStr(1, idStr.at(0));

          classIndex = sub2ind(10, 10, std::stoi(rowStr), std::stoi(colStr));
        }

        for (size_t j = 1, sampleId = 0; j < confusionClassification.n_cols; j++)
        {
          if (confusionClassification(classIndex, j) != 0)
          {
            arma::mat input = XTrain.col(
                confusionClassification(classIndex, j));
            input.reshape(28, 28);

            std::stringstream samplePropConfusion;
            samplePropConfusion << std::fixed << std::setprecision(2);
            samplePropConfusion << confusionProps(classIndex, j);

            std::string output = "{";
            output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
            output += "\"samplePropConfusion" + std::to_string(sampleId) + "\": \"" + samplePropConfusion.str() + "\",";
            std::string imageString = graphics::Mat2Image(arma::normalise(input) * 255);
            output += "\"sampleConfusion" + std::to_string(sampleId++) + "\": \"";
            output += websocketpp::base64_encode(imageString) + "\"";
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
        }

        m.unlock();
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
       * Get the continues index of the given row col.
       */
      size_t sub2ind(const size_t rows,
                     const size_t cols,
                     const size_t row,
                     const size_t col)
      {
         return row*cols+col;
      }

       /*
       * Send the current confusion matrix to all connected clients.
       */
      void SendConfusion(arma::Mat<int>& confusion)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream consufionString;
        for (size_t i = 0; i < confusion.n_elem; i++)
        {
          consufionString << confusion(i);

          if (i != (confusion.n_elem - 1))
            consufionString << ";";
        }

        std::stringstream precisionString;
        for (size_t i = 0; i < confusion.n_rows; i++)
        {
          double tp = confusion(i, i);
          double fp = arma::accu(confusion.row(i)) - tp;

          double precision = tp / (tp + fp);

          if (precision != precision)
            precision = 0;

          precisionString << std::fixed << std::setprecision(2);
          precisionString << precision << ";";
        }

        std::stringstream recallString;
        for (size_t i = 0; i < confusion.n_rows; i++)
        {
          double tp = confusion(i, i);
          double fn = arma::accu(confusion.col(i)) - tp;

          double recall = tp / (tp + fn);

          if (recall != recall)
            recall = 0;

          recallString << std::fixed << std::setprecision(2);
          recallString << recall << ";";
        }

        double accuracy = (double) arma::accu(confusion.diag()) /
            (double) arma::accu(confusion);

        if (accuracy != accuracy)
          accuracy = 0;

        std::stringstream accuracyString;
        accuracyString << std::fixed << std::setprecision(2);
        accuracyString << accuracy;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"precision\": \"" + precisionString.str() + "\",";
        output += "\"recall\": \"" + recallString.str() + "\",";
        output += "\"accuracy\": \"" + accuracyString.str() + "\",";
        output += "\"confusion\": \"" + consufionString.str() + "\"";
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
       * Send the current glimpse parameter to all connected clients.
       */
      void SendLocation(const arma::mat& input,
                        const arma::mat& locations,
                        const size_t inSize,
                        const size_t size,
                        const size_t depth,
                        const size_t scale)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        // Create local glimpse layer.
        GlimpseLayer<> localGlimpseLayer(1, 8, 3, 2);
        arma::cube localInput = arma::cube(input.n_rows, input.n_cols, 1);
        localInput.slice(0) = input;
        arma::cube localOutput;
        std::string glimpseStr = "";

        std::stringstream locationString;
        for (size_t l = 0; l < locations.n_cols; l++)
        {
          arma::mat location = locations.col(l);

          localGlimpseLayer.Location(location);
          localGlimpseLayer.Forward(localInput, localOutput);

          glimpseStr += GlimpseString(localOutput);
          GlimpseString(localOutput);

          size_t inputDepth = 1 / inSize;

          for (size_t inputIdx = 0, locactionIdx = 0; inputIdx < inSize; inputIdx++)
          {
            for (size_t depthIdx = 0, glimpseSize = size;
                depthIdx < depth; depthIdx++, glimpseSize *= scale, locactionIdx++)
            {
              size_t padSize = std::floor((glimpseSize - 1) / 2);

              size_t h = (input.n_rows + padSize * 2) - glimpseSize;
              size_t w = (input.n_cols + padSize * 2) - glimpseSize;

              size_t x = std::min(h, (size_t) std::max(0.0,
                  (location(0, inputIdx) + 1) / 2.0 * h));
              size_t y = std::min(w, (size_t) std::max(0.0,
                  (location(1, inputIdx) + 1) / 2.0 * w));

              locationString << x << ";" << y << ";";
            }
          }
        }

        // Remove the last ";".
        glimpseStr.pop_back();

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"locationSample\": \"" + locationString.str() + ":;" + glimpseStr + "\"";
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
       * Create glimpse activation string.
       */
      std::string GlimpseString(arma::cube& input)
      {
        if (connectionHandles.size() == 0)
        {
          std::string empty = "";
          return empty;
        }

        std::stringstream inputString;

        for (size_t i = 0; i < input.n_slices; i++)
        {
          arma::mat sample = input.slice(i);

          for (size_t j = 0; j < sample.n_elem; j++)
          {
            inputString << sample(j);
            inputString << ";";
          }
        }

        // Remove the last ";".
        std::string inputStr = inputString.str();


        return inputStr;
      }

      /*
       * Send the current input to all connected clients.
       */
      void SendInput(arma::mat& input)
      {
        if (connectionHandles.size() == 0)
          return;

        std::stringstream inputString;
        for (size_t i = 0; i < input.n_elem; i++)
        {
          inputString << input(i);

          if (i != (input.n_elem - 1))
            inputString << ";";
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"input\": \"" + inputString.str() + "\"";
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

      // The job state (running = 0, queued = 1, stopped = 2).
      int state;

      // The current error and the overall training error.
      double trainingError, trainingErrorcurrent;

      // The current error and the overall testing error.
      double testingError, testingErrorcurrent;

      // The current training sequence and the overall training sequence.
      size_t trainingSeqStart, trainingSeqCur;

      // The jon content.
      std::string content;

      // The dataset and the target.
      arma::mat YTrain, YTest;

      arma::mat XTrain, XTest;

      // The number of points in the dataset.
      arma::uword nPointsTrain, nPointsTest;

      // The gradient of the first  and second convolution layer.
      arma::cube gradient0, gradient1;

      // Locator network.
      LinearMappingLayer<>* linearLayer0;
      BiasLayer<>* biasLayer0;
      HardTanHLayer<>* hardTanhLayer0;
      ReinforceNormalLayer<>* reinforceNormalLayer0;
      HardTanHLayer<>* hardTanhLayer1;
      MultiplyConstantLayer<>* multiplyConstantLayer0;

      // Location sensor network.
      LinearLayer<>* linearLayer1;
      BiasLayer<>* biasLayer1;
      ReLULayer<>* rectifierLayer0;

      // Glimpse sensor network.
      GlimpseLayer<>* glimpseLayer0;
      LinearMappingLayer<>* linearLayer2;
      BiasLayer<>* biasLayer2;
      ReLULayer<>* rectifierLayer1;

      // Glimpse network.
      LinearLayer<>* linearLayer3;
      BiasLayer<>* biasLayer3;
      ReLULayer<>* rectifierLayer2;
      LinearLayer<>* linearLayer4;
      BiasLayer<>* biasLayer4;

      // Feedback network.
      LinearLayer<>* recurrentLayer0;
      BiasLayer<>* recurrentLayerBias0;

      // Start network.
      AdditionLayer<>* startLayer0;

      // Transfer network.
      ReLULayer<>* rectifierLayer3;

      // Classifier network.
      LinearLayer<>* linearLayer5;
      BiasLayer<>* biasLayer6;
      LogSoftmaxLayer<>* logSoftmaxLayer0;

      // Reward predictor network.
      ConstantLayer<>* constantLayer0;
      AdditionLayer<>* additionLayer0;

      size_t rho;
      size_t maxIterations;

      //! The current network parameter;
      arma::mat networkParameter;

      arma::mat meanSquaredGradient;

      //! Locally stored server object used for sending.
      server* s;

      //! List of connection handles.
      std::vector<connection_hdl> connectionHandles;

      //! Locally stored session id.
      int sessionid;

      bool predictionAvailable;

      arma::Mat<int> confusion;


      arma::Mat<int> confusionClassification;

      arma::mat confusionProps;

      bool sendConfusion;

      std::mutex m;

      bool reset;

      bool step;

      size_t sampleIndex;
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

        XTrain.load("/home/marcus/src/mlpack_work/build/mnist_large.csv");
        YTrain.load("/home/marcus/src/mlpack_work/build/mnist_large_target.csv");

        XTrain.reshape(28 * 28, 50000);

        XTest = XTrain;
        YTest = YTrain;
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
        else if(eventInfo.event == "step" && eventInfo.eventId >= 0)
        {
          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }

          state = jobQueue[uriInfo.id].State();

          jobQueue[uriInfo.id].Step() = true;

          if (state == 2)
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
        else if(eventInfo.event == "confusion" && eventInfo.eventId >= 0)
        {
          jobQueue[uriInfo.id].ConfusionSamples(eventInfo.eventId);
        }
        else if(eventInfo.event == "reset" && eventInfo.eventId >= 0)
        {
          jobQueue[uriInfo.id].Reset() = true;

          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }

          state = jobQueue[uriInfo.id].State();
          jobQueue[uriInfo.id].Reset() = true;

          if (state == 2)
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

      arma::mat XTrain, XTest;
      arma::mat YTrain, YTest;
      arma::uword nPointsTrain, nPointsTest;
  };
}