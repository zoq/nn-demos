#include <stdlib.h>
#include <iostream>
#include <string>
#include <mutex>
#include <boost/lexical_cast.hpp>

#include "fcgio.h"

// Maximum number of bytes allowed to be read from stdin.
const unsigned long STDIN_MAX = 1000000;

// Mutex used to synchronizing the access to the FCGX_GetParam function.
std::mutex requestContentMutex;

/**
 * Get the request content using the specified request.
 *
 * @param request - The request used to extract the request content.
 * @return The request content.
 */
std::string RequestContent(const FCGX_Request& request)
{
  std::string contentLengthString;
  {
    // Guard action on the static buffer using in the FCGX_GetParam function.
    requestContentMutex.lock();

    char *contentLengthCptr = FCGX_GetParam("CONTENT_LENGTH", request.envp);
    contentLengthString = contentLengthCptr;
  } // Unlock the mutex.

  size_t contentLength = 0;
  if (!contentLengthString.empty())
  {
    try {
      contentLength = boost::lexical_cast<size_t>(contentLengthString);
      if (contentLength > STDIN_MAX)
      {
        contentLength = STDIN_MAX;
      }
    }
    catch(boost::bad_lexical_cast const&)
    {
      std::cerr << "Can't Parse 'CONTENT_LENGTH='" << contentLengthString
                << "'. Consuming stdin up to " << STDIN_MAX << "\n";
      contentLength = STDIN_MAX;
    }
  }
  else
  {
    // Do not read from stdin if CONTENT_LENGTH is missing or unparsable.
    contentLength = 0;
  }

  char * contentBuffer = new char[contentLength];
  std::cin.read(contentBuffer, contentLength);
  contentLength = std::cin.gcount();

  // Chew up any remaining stdin - this shouldn't be necessary
  // but is because mod_fastcgi doesn't handle it correctly.

  // ignore() doesn't set the eof bit in some versions of glibc++
  // so use gcount() instead of eof()...
  do std::cin.ignore(1024); while (std::cin.gcount() == 1024);

  std::string content(contentBuffer, contentLength);
  delete [] contentBuffer;
  return content;
}
