/**
 * @file config.hpp
 * @author Marcus Edel
 *
 * Miscellaneous settings.
 */

#include <iostream>
#include <string>

namespace config {
  // The deployment path.
  const static std::string path = "rmva";

  // The websocket port.
  const static int port = 9973;

  // The deployment url.
  const static std::string url = "http://kurg.org/rmva/index.html";

  // The train dataset to be loaded.
  const static std::string trainDataset = "mnist_train.csv";

  // The test dataset to be loaded.
  const static std::string testDataset = "mnist_test.csv";
}