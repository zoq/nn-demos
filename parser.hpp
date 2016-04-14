/**
 * @file parser.hpp
 * @author Marcus Edel
 *
 * Miscellaneous parser routines.
 */

#include <iostream>
#include <string>
#include <stdlib.h>
#include <exception>

namespace parser {
  // Info structure used to store the parsed information.
  struct info
  {
    int id;
    std::string event;
    int eventId;
  };

  // The event vector contains all events that are parsed by the parser.
  std::vector<std::string> events = {"weight",
                                     "gradient",
                                     "input",
                                     "prediction",
                                     "info",
                                     "start",
                                     "state",
                                     "confusion",
                                     "reset",
                                     "step"};

  /*
   * Cuts off the delimiter (default) and everything that follows.
   *
   * @param text - The input text.
   * @param delimiter - The delimiter that should be cut off.
   * @keepDelimiter - Keep the delimiter and cut everything that follows after
   *                  the delimiter.
   * @return The parsed text.
   */
  static inline std::string TailSlice(std::string& text,
                                      std::string delimiter,
                                      bool keepDelimiter = false)
  {
    size_t delimiterLocation = text.find(delimiter);

    std::string output = "";
    if (delimiterLocation < std::string::npos)
    {
        size_t start = keepDelimiter ?
            delimiterLocation : delimiterLocation + delimiter.length();

        size_t end = text.length() - start;
        output = text.substr(start, end);
        text = text.substr(0, delimiterLocation);
    }

    return output;
  }

  /*
   * Cuts off the delimiter (default) and everything that precedes.
   *
   * @param text - The input text.
   * @param delimiter - The delimiter that should be cut off.
   * @return The parsed text.
   */
  static inline std::string HeadSlice(std::string &text,
                                      std::string delimiter)
  {
      auto delimiterLocation = text.find(delimiter);

      std::string output = "";
      if (delimiterLocation < std::string::npos)
      {
          output = text.substr(0, delimiterLocation);
          text = text.substr(delimiterLocation + delimiter.length(),
              text.length() - (delimiterLocation + delimiter.length()));
      }

      return output;
  }

  /*
   * Function to extract the event id (uri := domain.tld/{event}={event id}.
   *
   * @param uri - The input uri.
   * @param event - The event that should be used to extract the id.
   * @return The event id.
   */
  static inline int ExtractEventID(std::string uri, std::string event)
  {
    int id = -1;
    std::string idString = HeadSlice(uri, event + "=");

    if (idString.length() > 0)
    {
      TailSlice(uri, "&");

      try
      {
        id = atoi(uri.c_str());
      }
      catch (std::exception e)
      {
        id = -1;
      }
    }

    return id;
  }

  /*
   * Function to extract the job id (uri := domain.tld/id={job id} using the
   * uri as input.
   *
   * @param uri - The input uri.
   * @return The job id.
   */
  static inline int ExtractID(std::string uri)
  {
    return ExtractEventID(uri, "id");
  }

  /*
   * Function to extract the first event (uri := domain.tld/{event}={job id}
   * using the uri as input. All events are defined in the events vector.
   *
   * @param uri - The input uri.
   * @return The event string.
   */
  static inline std::string ExtractEvent(std::string uri)
  {
    for (std::string event : events)
    {
      std::string eventString = HeadSlice(uri, event + "=");
      if (eventString.length() > 0)
      {
        return event;
      }
    }

    return "";
  }

  /*
   * Extract the job id and the first event and event id from the specified uri.
   *
   * @param uri - The input uri.
   * @param urlInfo - The info object used to store the information.
   */
  static inline void parseURI(const std::string& uri, info& urlInfo)
  {
    urlInfo.id = ExtractID(uri);
    urlInfo.event = "";
    urlInfo.eventId = -1;

    if (urlInfo.id >= 0)
    {
      urlInfo.event = ExtractEvent(uri);

      if (urlInfo.event != "")
      {
        urlInfo.eventId = ExtractEventID(uri, urlInfo.event);
      }
    }
  }

}