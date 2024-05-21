import axios from 'axios';
import { CopilotKitResponse, CopilotKitServiceAdapter } from "../types/service-adapter";
import { limitUnifyMessagesToTokenCount, maxTokensForUnifyModel } from "../utils/unify";

const DEFAULT_MODEL = "mistral-7b-instruct-v0.2@fireworks-ai";
const UNIFY_API_URL = "https://api.unify.ai/v0/inference";

export interface UnifyAdapterParams {
  apiKey: string;
  model?: string;
}

export class UnifyAdapter implements CopilotKitServiceAdapter {
  private model: string = DEFAULT_MODEL;
  private apiKey: string;

  constructor(params: UnifyAdapterParams) {
    if (!params.apiKey) {
      throw new Error("API key is required for UnifyAdapter");
    }
    this.apiKey = params.apiKey;
    if (params.model) {
      this.model = params.model;
    }
  }

  async getResponse(forwardedProps: any): Promise<CopilotKitResponse> {
    // copy forwardedProps to avoid modifying the original object
    forwardedProps = { ...forwardedProps };

    // Remove tools if there are none to avoid Unify API errors
    // when sending an empty array of tools
    if (forwardedProps.tools && forwardedProps.tools.length === 0) {
      delete forwardedProps.tools;
    }

    const messages = limitUnifyMessagesToTokenCount(
      forwardedProps.messages || [],
      forwardedProps.tools || [],
      maxTokensForUnifyModel(forwardedProps.model || this.model),
    );

    // remove message.function_call.scope if it's present.
    // scope is a field we inject as a temporary workaround (see elsewhere), which unify doesn't understand
    messages.forEach((message) => {
      if (message.function_call?.scope) {
        delete message.function_call.scope;
      }
    });

    try {
      const response = await axios.post(
        UNIFY_API_URL,
        {
          model: this.model.split('@')[0], // Extract the model name
          provider: this.model.split('@')[1], // Extract the provider
          arguments: {
            ...forwardedProps,
            stream: true,
            messages: messages as any,
          },
        },
        {
          headers: {
            'Authorization': `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
            'accept': 'application/json',
          },
          responseType: 'stream',
        }
      );

      return { stream: response.data };
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Error fetching response from Unify API: ${error.message}`);
      } else {
        throw new Error('An unknown error occurred while fetching response from Unify API');
      }
    }
  }
}