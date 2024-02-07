import os
import pdb
import time
import openai
import json
import re

openai.api_key = os.environ["OPENAI_API_KEY"]

class LangModel():
    # Note that this writes to the cache even if read_cache is False. That flag is only used for reading.
    # To avoid any cache connection, just do not pass a cache_path.
    def __init__(self, read_cache=True, cache_path=""):
        self.check_cache = read_cache
        self.cache_path = cache_path
        if cache_path:
            self.cache = json.load(open(cache_path, "r"))

    def submit_prompt(self, prompt, temperature=0.0, silent=False):
        if self.cache_path and self.check_cache and prompt in self.cache.keys():
            if not silent:
                print(f'Using response found in cache for prompt: "{prompt}"')
            completion = self.cache[prompt]
            if not silent:
                print(f'Returning response: "{completion}"')
            return completion
        else:
            if not silent:
                print(f'Submitting prompt to GPT: "{prompt}"')
            max_len = 5000
            if len(prompt) > max_len:
                raise Exception(f"Prompt too long (length: {len(prompt)}). Max length is {max_len}.")

            tries = 3
            while tries > 0:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[{"content": prompt, "role": "user"}],
                        temperature=temperature,
                        max_tokens=200,
                    )
                    break
                except (openai.error.APIError, openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                    tries -= 1
                    if tries == 0:
                        raise e
                    time.sleep(0.5)

            completion = response["choices"][0]["message"]["content"] # type: ignore
            if self.cache_path:
                self.cache[prompt] = completion
                json.dump(self.cache, open(self.cache_path, "w"), indent=4)
            if not silent:
                print(f'Returning response: "{completion}"')
            return completion

    def get_principal_noun(self, caption):
        prompt = f'Suppose that you have an image caption describing a scene. What is the name of the most important object in this scene? Please answer only with one word, the name of the object. Caption: "{caption}"'
        response = self.submit_prompt(prompt)
        response = response.lower()
        response = response.replace(".", "")
        return response

    def get_movable_obj_idx(self, user_instr, obj_captions):
        prompt = f'Suppose that you are a robot. There are some objects in the scene. The user gives you an instruction. Decide which one object the user wants the robot to move. Do not include any objects which should remain unmoved (e.g. containers). Below, a description is given for each of the objects. You must answer with only one number, the index of the object which should be moved.\n'
        prompt += f'User instruction: "{user_instr}"\n'
        assert obj_captions[0] == "__background__"
        for i, caption in enumerate(obj_captions[1:]): # Skip background
            prompt += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(prompt)
        movable_idx = int(re.findall(r'\d+', response)[0])
        return movable_idx

    def get_relevant_obj_idxs(self, scene_caption, obj_captions, movable_obj_idx):
        prompt = f'Suppose that you are a robot. You are given a caption of a scene. Below, you are also given some object descriptions. For each object description, determine whether it is a distractor object. Return a separate line for each object containing Yes or No, where Yes means that it is a distractor. A distractor object is one which cannot possibly be one of the objects mentioned in the scene caption. Be careful that the object descriptions are based on low-quality images where the text is not easily identified, so ignore that part of the object descriptions. If the object description could plausibly describe an object in the scene, you must return No. Each line in the response should have the format: Object <number>: Yes/No. But if none of the objects in the scene are distractors, the final line should just be one word: "None".\n'
        prompt += f'Scene caption: "{scene_caption}"\n'
        assert obj_captions[0] == "__background__"

        # Temporarily swap object at idx 1 with movable object, so that LLM sees movable first. Autoregression can be weird.
        obj_captions = obj_captions.copy()
        temp = obj_captions[1]
        obj_captions[1] = obj_captions[movable_obj_idx]
        obj_captions[movable_obj_idx] = temp

        for i, caption in enumerate(obj_captions[1:]): # Skip background
            prompt += f'Object {i + 1}: "{caption}"\n'

        response = self.submit_prompt(prompt)
        decisions = response.split("\n")

        if decisions[-1] == "None":
            return range(1, len(obj_captions))

        relevant_idxs = [movable_obj_idx] # Movable always relevant
        for i, decision in enumerate(decisions):
            # Skip movable
            if i == 0:
                continue
            if 'Yes' not in decision:
                 # Add 1 to account for background
                 # This undoes the temporary swap above.
                relevant_idx = 1 if i + 1 == movable_obj_idx else i + 1
                relevant_idxs.append(relevant_idx)
        assert len(decisions) + 1 == len(obj_captions), "Error: LLM returned wrong number of decisions for distractor status for objects"
        return relevant_idxs

    # Aggregate captions across views for single object.
    def aggregate_captions_for_obj(self, captions, silent=True):
        prompt = f'Suppose we have captured many images of an object across different views. For each view, we have asked a network to caption the image. Some captions may be wrong, and there may be some other objects in view accidentally (e.g. inside or on top of the main object) which you must ignore. Please aggregate the caption information from across views, and write a caption which best describes the main object being captured. If the object can be a couple of things, mention them both.\n'
        prompt += f'List of captions:\n'
        for caption in captions:
            prompt += f'"{caption}"\n'

        response = self.submit_prompt(prompt, silent=silent)
        return response

    def parse_instr(self, user_instr):
        prompt = f'Suppose you are a robot. You are given an instruction from a user. First, you need to extract the goal caption from the prompt. This is a description of the desired state after the user instruction has been executed. E.g. if the instruction is "shove the X under Y", the goal caption would be "an X under a Y". Also, you should extract a normalising caption from the goal caption. This will list the objects mentioned in the goal caption but without any spatial relations. Your first returned line should be the goal caption (the line should begin with "Goal caption: "), and the second line should be the normalising caption (the line should begin with "Normalising caption: "). No quotation marks needed. E.g. if the goal caption is "an X under a Y", then the normalising caption would be "an X and a Y". If the goal caption is "big Xs in the style of something", then the normalising caption is just "big Xs". However, you should keep spatial relations if they refer to a table, because objects will always be above table level. E.g. if the goal caption is "Xs arranged in a grid on a plastic table", then the normalising caption would be "Xs on a plastic table".\n'
        prompt += f'User instruction: "{user_instr}"\n'
        response = self.submit_prompt(prompt)
        goal_caption, norm_caption = response.split("\n")
        goal_caption = goal_caption.replace("Goal caption: ", "")
        norm_caption = norm_caption.replace("Normalising caption: ", "")
        return goal_caption, norm_caption