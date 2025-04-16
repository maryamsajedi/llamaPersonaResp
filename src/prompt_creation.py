#define a class "prompt"
class infoPrompt:
    def __init__(self, scenario_description, input_elements, task_definition, input_format, instruction, output_format, command, output_template):
        self.scenario_description = scenario_description
        self.input_elements = input_elements
        self.task_definition = task_definition
        self.input_format = input_format
        self.instruction = instruction
        self.output_format = output_format
        self.command = command
        self.output_template = output_template


class PromptLlamaMPC:
    def __init__(self, info):
        self.scenario_description = info.scenario_description
        self.input_elements = info.input_elements
        self.task_definition = info.task_definition
        self.input_format = info.input_format
        self.instruction = info.instruction
        self.output_format = info.output_format
        self.command = info.command
        self.output_template = info.output_template

    def create_system_instruction(self, conversation=False,  profile=False, response_selection=False):
        scenario_description = self.scenario_description
        input_elements = self.input_elements['general_statement']
        task_definition = ""
        input_format = ""
        instruction = ""
        output_format = ""

        if conversation:
            input_elements = input_elements + "\n\n" + self.input_elements['conversation']
            input_format = input_format + "\n\n" + self.input_format['conversation']

        if profile:
            input_elements = input_elements + "\n\n" + self.input_elements['description']
            input_format = input_format + "\n\n" + self.input_format['description']



        input_format = input_format[2:]

        if response_selection:
            task_definition = self.task_definition['response_selection']
            instruction = self.instruction['response_selection']
            output_format = self.output_format['response_selection']
       

        system_instruction = ("<<SYS>>\n\n" +
                              scenario_description + "\n\n" +
                              input_elements + "\n\n" +
                              task_definition + "\n\n" +
                              input_format + "\n\n" +
                              instruction + "\n\n" +
                              output_format + "\n\n<</SYS>>")

        return system_instruction

    def create_input(self, item, conversation=False, profile=False):

        item_input = ""

        if conversation:
            item_input = item_input + "[CONVERSATION]\n" + item['conversation'] + "\n[/CONVERSATION]\n\n"

        if profile:
            item_input = item_input + "[PROFILE]\n" + item['profile'] + "\n[/PROFILE]\n\n"


        return item_input

    def response_selection(self, conversation=False, profile=False, item = ""):

        system_instruction = self.create_system_instruction(conversation=conversation, profile=profile, response_selection=True)

        if item == "":
            return system_instruction

        else:
            item_input = self.create_input(item, conversation=conversation, profile=profile)
            prompt = "<s>[INST]" + system_instruction + "\n\n" + item_input + self.command['response_selection'] + "\n\n" + "[/INST]\n\n" + self.output_template['response_selection']

            return prompt, system_instruction
